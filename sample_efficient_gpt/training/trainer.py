import math
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from jaxtyping import Int
from tqdm.auto import tqdm

from sample_efficient_gpt.transformer import Transformer
from sample_efficient_gpt.training import (
    load_checkpoint,
    save_checkpoint,
    MemoryMappedDataset,
    AdamW,
    Muon,
    Lion,
    get_cosine_lr,
    get_wsd_lr,
    clip_grad_norm_,
    cross_entropy,
)
from sample_efficient_gpt.config_schema import Config
from sample_efficient_gpt.utils.logger import logger
from sample_efficient_gpt.utils.config_tools import load_config, apply_overrides

from sample_efficient_gpt.tokenizer import Tokenizer
from sample_efficient_gpt.evals.mbpp_eval import SimpleMBPPEvaluator


torch.set_float32_matmul_precision("high")


def mem(tag):
    alloc = torch.cuda.memory_allocated() / 1e9
    resv = torch.cuda.memory_reserved() / 1e9
    logger.info(f"{tag}: allocated={alloc:.2f} GB, reserved={resv:.2f} GB")


class Trainer:
    """
    class that takes cfg: Config and runs training
    """

    def __init__(
        self,
        cfg: Config | None = None,
        load_from: str | None = None,
        load_components: str = "all",
        wandb: Any | None = None,
        **cfg_overrides: dict,
    ):
        if cfg is None:
            assert load_from is not None, "you must load from checkpoint if cfg is None"
            self.cfg = load_config(torch.load(load_from, map_location="cpu")["config"])
            self.cfg = apply_overrides(self.cfg, cfg_overrides)
            logger.info(f"Loading from the checkpoint, device={self.cfg.trainer.device}")
        else:
            logger.info("Loading from config")
            self.cfg = cfg
        self.model = Transformer(
            self.cfg.model.n_layers,
            self.cfg.model.vocab_size,
            self.cfg.model.d_model,
            self.cfg.model.n_heads,
            self.cfg.model.d_ff,
            attn_qknorm=self.cfg.model.attn_qknorm,
            attn_val_residual=self.cfg.model.attn_val_residual,
            attn_gating=self.cfg.model.attn_gating,
            layernorm_scaling=self.cfg.model.layernorm_scaling,
            theta=self.cfg.model.theta,
            device=self.cfg.trainer.device,
            # always keep master weights in fp32
            dtype=torch.float32,
            weight_tying=self.cfg.model.weight_tying,
        )
        layer_params = (
            4 * self.cfg.model.d_model**2
            + 2 * self.cfg.model.d_model
            + 2 * self.cfg.model.d_model * self.cfg.model.d_ff
        )
        total_params = self.cfg.model.n_layers * layer_params + self.cfg.model.d_model * self.cfg.model.vocab_size
        logger.info(
            f"Created a model with {layer_params / 1e6:.2f}M layer and {total_params / 1e6:.1f}M total non-emb params"
        )
        self.model.to(self.cfg.trainer.device)
        self.model.compile()
        if load_components == "all":
            self._init_optimizers()

            logger.info(f"Model is created with hparams {self.cfg.model}")
            self.train_dataset = MemoryMappedDataset(
                self.cfg.data.train_path,
                self.cfg.data.context_length,
                torch.device(self.cfg.trainer.device),
                self.cfg.data.seed,
            )
            self.val_dataset = MemoryMappedDataset(
                self.cfg.data.validation_path,
                self.cfg.data.context_length,
                torch.device(self.cfg.trainer.device),
                self.cfg.data.seed,
            )
            self.save_dir = Path(self.cfg.trainer.save_dir)
            self.save_dir.mkdir(exist_ok=True, parents=True)
        else:
            self.optimizers = None
        self.iteration = 0
        self.wandb = wandb
        load_from = load_from or self.cfg.trainer.load_from
        if load_from is not None:
            self.load_state(load_from)

        # aux scaler for bf16 if used
        if self.cfg.trainer.dtype == "bfloat16":
            self.scaler = torch.amp.grad_scaler.GradScaler()

        tokenizer_path = "/home/george/cs336_solutions/assignment1-basics/tokenizer/owt"
        tokenizer = Tokenizer.from_files(
            Path(tokenizer_path) / "vocab.pickle",
            Path(tokenizer_path) / "merges.pickle",
            special_tokens=["<|endoftext|>"],
        )
        eos_token_id = tokenizer.encode(tokenizer.special_tokens[0])[0]
        self.generate_fn = lambda prompt: tokenizer.decode(
            self.generate(
                torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).to(self.cfg.trainer.device), eos_token_id
            )[0]
            .cpu()
            .tolist()
        )
        self.mbbp_evaluator = SimpleMBPPEvaluator(dataset="lite", timeout=8.0)

    def _init_optimizers(self):
        if self.cfg.optim.use_muon:
            one_d_params = [
                p for n, p in self.model.named_parameters() if p.ndim < 2 or "embedding" in n or "lm_head" in n
            ]
            # Implement muP scaling, for embedding it's sqrt(d), for out it's 0.5
            for n, p in self.model.named_parameters():
                if "embedding" in n:
                    setattr(p, "lr_mul", self.cfg.model.d_model**0.5)
                elif "lm_head" in n:
                    setattr(p, "lr_mul", 0.5)
            two_d_params = [
                p
                for n, p in self.model.named_parameters()
                if p.ndim >= 2 and "embedding" not in n and "lm_head" not in n
            ]
            if self.cfg.optim.use_lion:
                self.optimizer1 = Lion(
                    one_d_params,
                    lr=self.cfg.optim.lr,
                    betas=self.cfg.optim.betas,
                    weight_decay=self.cfg.optim.wd,
                )
            else:
                self.optimizer1 = AdamW(
                    one_d_params,
                    lr=self.cfg.optim.lr,
                    betas=self.cfg.optim.betas,
                    weight_decay=self.cfg.optim.wd,
                )
            if self.cfg.optim.muon_lr is None:
                muon_lr = self.cfg.optim.lr
            else:
                muon_lr = self.cfg.optim.muon_lr
            if self.cfg.optim.muon_wd is None:
                muon_wd = self.cfg.optim.wd
            else:
                muon_wd = self.cfg.optim.muon_wd
            self.optimizer2 = Muon(
                two_d_params,
                lr=muon_lr,
                momentum=self.cfg.optim.betas[0],
                weight_decay=muon_wd,
            )
            self.optimizers = [self.optimizer1, self.optimizer2]
        else:
            self.optimizer = AdamW(
                self.model.parameters(),
                lr=self.cfg.optim.lr,
                betas=self.cfg.optim.betas,
                weight_decay=self.cfg.optim.wd,
            )
            self.optimizers = [self.optimizer]

    @property
    def tokens_processed(self):
        return self.iteration * self.cfg.data.batch_size * self.cfg.data.context_length

    def load_state(self, path: Path):
        logger.info(f"Loading state from {str(path)}")
        self.iteration = load_checkpoint(path, self.model, self.optimizers, device=self.cfg.trainer.device)

    def save_state(self):
        logger.info(f"Saving training state at iter={self.iteration}")
        save_checkpoint(
            self.save_dir / f"{self.iteration}.pt",
            self.cfg,
            self.model,
            self.optimizers,
            self.iteration,
        )

    def log(self, **data):
        for k, v in data.items():
            if "log" not in k:
                logger.info(f"{k}: {v}")
        if self.wandb is not None:
            self.wandb.log(data)

    def generate(
        self,
        prompt: Int[Tensor, "bs seq"],
        eos_token_id: int,
        top_p: float = 1.0,
        temperature: float = 0.0,
        max_steps: int = 256,
    ):
        """
        Perform decoding with nucleous sampling and temperature
        """
        return self.model.generate(prompt, eos_token_id, top_p=top_p, temperature=temperature, max_steps=max_steps)

    def validate(self):
        mem("before validation")
        self.model.eval()
        val_iters = 0
        val_loss_epoch = torch.zeros((), device=self.model.embedding.weight.data.device, dtype=torch.float32)
        for inputs, targets in tqdm(
            self.val_dataset.get_iterator(self.cfg.data.val_batch_size),
            total=len(self.val_dataset) // (self.cfg.data.val_batch_size * self.cfg.data.context_length),
            desc="Running validation",
        ):
            with (
                torch.inference_mode(),
                torch.autocast("cuda", enabled=self.cfg.trainer.dtype == "bfloat16"),
            ):
                logits, _ = self.model(inputs)
                val_loss, _ = cross_entropy(logits, targets)
                val_loss_epoch += val_loss.to(torch.float32)
                val_iters += 1
        mem("after validation")
        val_loss_epoch = (val_loss_epoch / val_iters).item()

        metrics = self.mbbp_evaluator.evaluate_with_generate(self.generate_fn, num_samples=1, mode="full")
        logger.info(metrics)

        return {
            "val_loss": val_loss_epoch,
            "val_perplexity": math.exp(val_loss_epoch),
            "mbpp_lite_pass_1": metrics["pass_at_k"]["base"]["pass@1"]
        }

    def _get_lr(self, lr, lr_min):
        if self.cfg.optim.scheduler == "cosine":
            iter_lr = get_cosine_lr(
                self.iteration,
                lr,
                lr_min,
                self.cfg.optim.warmup_steps,
                self.cfg.optim.cosine_steps,
            )
        elif self.cfg.optim.scheduler == "wsd":
            iter_lr = get_wsd_lr(
                self.iteration,
                lr,
                lr_min,
                self.cfg.optim.warmup_steps,
                self.cfg.optim.stable_steps,
                self.cfg.optim.decay_steps,
            )
        else:
            raise ValueError("unrecognized optim scheduler")
        return iter_lr

    def _set_lr(self):
        """
        Sets learning rate according to the learning rate schedule

        Works for both optimizers in a separate fashion, so AdamW receives smaller lr
        + support wd decay for Muon
        """
        iter_lr = self._get_lr(self.cfg.optim.lr, self.cfg.optim.lr * self.cfg.optim.lr_min_coeff)
        for pg in self.optimizers[0].param_groups:
            pg["lr"] = iter_lr

        if self.cfg.optim.use_muon:
            # second optimizer is muon
            iter_lr = self._get_lr(self.cfg.optim.muon_lr, self.cfg.optim.muon_lr * self.cfg.optim.lr_min_coeff)
            if self.cfg.optim.muon_wd_min is not None:
                iter_wd = self._get_lr(self.cfg.optim.muon_wd, self.cfg.optim.muon_wd_min)
            else:
                iter_wd = None

            for pg in self.optimizers[1].param_groups:
                # momentum warmup for fixed 300 steps
                momentum = self.cfg.optim.betas[0]
                pg["momentum"] = 0.85 + min(max(self.iteration, 1), 300) / 300 * (momentum - 0.85)
                pg["lr"] = iter_lr
                if iter_wd is not None:
                    pg["wd"] = iter_wd
            return iter_lr
        return iter_lr

    def train_step(self, inputs, targets):
        self.model.train()
        iter_lr = self._set_lr()

        z_loss_weight = self.cfg.trainer.z_loss_weight

        with torch.autocast("cuda", enabled=self.cfg.trainer.dtype == "bfloat16"):
            logits, prenorm_activation_norms = self.model(inputs)
            loss, z_loss = cross_entropy(logits, targets)
        if self.cfg.trainer.gradient_accumulation_steps > 1:
            loss /= self.cfg.trainer.gradient_accumulation_steps
            z_loss /= self.cfg.trainer.gradient_accumulation_steps

        if self.cfg.trainer.dtype == "bfloat16":
            self.scaler.scale(loss + z_loss_weight * z_loss).backward()
        else:
            (loss + z_loss_weight * z_loss).backward()

        update_ratio = 0.0
        if self.iteration % self.cfg.trainer.gradient_accumulation_steps == 0:
            if self.cfg.trainer.dtype == "bfloat16":
                for opt in self.optimizers:
                    self.scaler.unscale_(opt)
                grad_norm = clip_grad_norm_(self.model.parameters(), self.cfg.trainer.max_grad_norm)
                for opt in self.optimizers:
                    self.scaler.step(opt)
                self.scaler.update()
            else:
                grad_norm = clip_grad_norm_(self.model.parameters(), self.cfg.trainer.max_grad_norm)
                for opt in self.optimizers:
                    opt.step()
            for opt in self.optimizers:
                opt.zero_grad(set_to_none=True)
        else:
            grad_norm = torch.tensor(0.0)

        parameter_norms = {
            k: p.data.detach().norm() for k, p in self.model.named_parameters() if "attn" in k or "ffn" in k
        }
        train_loss = loss.detach()
        z_loss = z_loss.detach()
        return {
            "train_loss": train_loss * self.cfg.trainer.gradient_accumulation_steps,
            "z_loss": z_loss * self.cfg.trainer.gradient_accumulation_steps,
            "grad_norm": grad_norm,
            "lr": iter_lr,
            "update_ratio": update_ratio,
            "prenorm_activation_norms": prenorm_activation_norms,
            "parameter_norms": parameter_norms,
        }

    def postprocess_step_stats(self, step_stats):
        return {
            "train_loss": step_stats["train_loss"].item(),
            "z_loss": step_stats["z_loss"].item(),
            "grad_norm": step_stats["grad_norm"].item(),
            "lr": step_stats["lr"],
            "update_ratio": step_stats["update_ratio"],
            "prenorm_activation_norms": step_stats["prenorm_activation_norms"].cpu().tolist(),
            "parameter_norms": {k: v.item() for k, v in step_stats["parameter_norms"].items()},
        }

    def train(self):
        logger.info("Starting training loop")
        while self.iteration < self.cfg.trainer.max_steps:
            if self.iteration % self.cfg.trainer.save_every == 0:
                self.save_state()
            if self.iteration > 0 and self.iteration % self.cfg.trainer.val_every == 0:
                val_metrics = self.validate()
                self.log(**val_metrics)

            inputs, targets = self.train_dataset.get_batch(self.cfg.data.batch_size)
            t0 = time.monotonic()
            step_stats = self.train_step(inputs, targets)
            t1 = time.monotonic()
            if self.iteration % self.cfg.trainer.log_every == 0:
                logger.info(f"Train iteration {self.iteration}")
                # call item and introduce sync
                step_stats = self.postprocess_step_stats(step_stats)
                prenorm_log = {f"log/block{i}_prenorm": v for i, v in enumerate(step_stats["prenorm_activation_norms"])}
                pnorm_log = {f"log/{k}_norm": v for k, v in step_stats["parameter_norms"].items()}
                self.log(
                    **{
                        "train/loss": step_stats["train_loss"],
                        "train/z_loss": step_stats["z_loss"],
                        "train/grad_norm": step_stats["grad_norm"],
                        "train/learning_rate": step_stats["lr"],
                        "train/update_ratio": step_stats["update_ratio"],
                        "train/step": self.iteration,
                        "train/step_time": t1 - t0,
                        "train/tokens_processed": self.tokens_processed,
                        **prenorm_log,
                        **pnorm_log,
                    }
                )
            self.iteration += 1
        self.save_state()
        val_metrics = self.validate()
        self.log(**val_metrics)
