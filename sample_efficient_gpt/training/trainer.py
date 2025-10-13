import math
import time
from pathlib import Path
from typing import Any
from contextlib import nullcontext

import torch
import torch.distributed as dist
from torch import Tensor
from jaxtyping import Int
from tqdm.auto import tqdm
from transformers import PreTrainedTokenizerFast

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

from sample_efficient_gpt.training.distributed import DDP, ShardedOptimizer
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

        self.is_distributed = dist.is_initialized()
        if self.is_distributed:
            self.world_size = dist.get_world_size()
            self.rank = dist.get_rank()
        else:
            self.world_size = 1
            self.rank = 0

        if load_components == "all":
            self._init_optimizers()

            logger.info(f"Model is created with hparams {self.cfg.model}")
            self.train_dataset = MemoryMappedDataset(
                self.cfg.data.train_path,
                self.cfg.data.context_length,
                torch.device(self.cfg.trainer.device),
                self.cfg.data.seed,
                world_size=self.world_size,
                rank=self.rank,
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
            self.model.eval()
            self.optimizers = None

        if self.is_distributed:
            self.model = DDP(self.model)

        self.iteration = 0
        self.wandb = wandb
        load_from = load_from or self.cfg.trainer.load_from
        if load_from is not None:
            self.load_state(load_from)

        # aux scaler for bf16 if used
        if self.cfg.trainer.dtype == "bfloat16":
            self.scaler = torch.amp.grad_scaler.GradScaler()

        tokenizer_path = self.cfg.data.tokenizer_path
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        self.eos_token_id = list(self.tokenizer.added_tokens_decoder)[0]
        self.mbbp_evaluator = SimpleMBPPEvaluator(dataset="lite", timeout=8.0)

        # Use per token bytes length for normalization in loss
        # self.id2byte_len = torch.tensor(
        #     [
        #         len(self.tokenizer.decode([i], clean_up_tokenization_spaces=False).encode("utf-8"))
        #         for i in range(self.tokenizer.vocab_size)
        #     ],
        #     device=self.cfg.trainer.device,
        # )
        self.id2byte_len = None
        # these are needed for fair comparison with non-superbpe
        if "superbpe" in self.cfg.data.tokenizer_path:
            self.train_loss_multiplier = 0.7985
            self.val_loss_multiplier = 0.8013
        else:
            self.train_loss_multiplier = 1
            self.val_loss_multiplier = 1

        logger.info(
            f"Init [rank={self.rank}]: max_steps={self.cfg.trainer.max_steps}, start_iter={self.iteration}, "
            f"val_every={self.cfg.trainer.val_every}, save_every={self.cfg.trainer.save_every}, "
            f"world_size={self.world_size}, grad_accum={self.cfg.trainer.gradient_accumulation_steps}, "
            f"batch_size={self.cfg.data.batch_size}",
        )

    @property
    def rank_zero_only(self):
        target_log_rank = 0
        for i in range(self.world_size):
            if "RTX 5090" in torch.cuda.get_device_name(i):
                target_log_rank = i
        return not self.is_distributed or (self.is_distributed and self.rank == target_log_rank)

    def _init_optimizers(self):
        optimizer_kwargs = dict(
            lr=self.cfg.optim.lr,
            betas=self.cfg.optim.betas,
            weight_decay=self.cfg.optim.wd,
        )
        if self.cfg.optim.use_lion:
            optimizer_cls = Lion
        else:
            optimizer_cls = AdamW
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

            one_d_optimizer_kwargs = optimizer_kwargs
            if self.is_distributed:
                self.optimizer1 = ShardedOptimizer(one_d_params, optimizer_cls, **one_d_optimizer_kwargs)
            else:
                self.optimizer1 = optimizer_cls(one_d_params, **one_d_optimizer_kwargs)

            if self.cfg.optim.muon_lr is None:
                muon_lr = self.cfg.optim.lr
            else:
                muon_lr = self.cfg.optim.muon_lr
            if self.cfg.optim.muon_wd is None:
                muon_wd = self.cfg.optim.wd
            else:
                muon_wd = self.cfg.optim.muon_wd

            two_d_optimizer_kwargs = dict(
                lr=muon_lr,
                momentum=self.cfg.optim.betas[0],
                weight_decay=muon_wd,
            )

            if self.is_distributed:
                self.optimizer2 = ShardedOptimizer(two_d_params, Muon, **two_d_optimizer_kwargs)
            else:
                self.optimizer2 = Muon(two_d_params, **two_d_optimizer_kwargs)
            self.optimizers = [self.optimizer1, self.optimizer2]
        else:
            if self.is_distributed:
                self.optimizer = ShardedOptimizer(one_d_params, optimizer_cls, **optimizer_kwargs)
            else:
                self.optimizer = optimizer_cls(one_d_params, **one_d_optimizer_kwargs)
            self.optimizers = [self.optimizer]

    @property
    def tokens_processed(self):
        return self.iteration * self.cfg.data.batch_size * self.cfg.data.context_length

    def load_state(self, path: Path):
        logger.info(f"Loading state from {str(path)}")
        self.iteration = load_checkpoint(
            path,
            self.model.module if self.is_distributed else self.model,
            self.optimizers,
            device=self.cfg.trainer.device,
        )

    def save_state(self):
        logger.info(f"Saving training state at iter={self.iteration}")
        save_checkpoint(
            self.save_dir / f"{self.iteration}.pt",
            self.cfg,
            self.model.module if self.is_distributed else self.model,
            self.optimizers,
            iteration=self.iteration,
            run_id=self.wandb.id if self.wandb is not None else None,
        )

    def log(self, data, rank_zero=True):
        if (not rank_zero) or self.rank_zero_only:
            for k, v in data.items():
                if "log" not in k:
                    logger.info(f"{k}: {v}")
            if self.wandb is not None:
                self.wandb.log(data, step=self.iteration)

    def generate(
        self,
        prompt: str,
        eos_token_id: int,
        top_p: float = 1.0,
        temperature: float = 0.0,
        max_steps: int = 256,
        eos_prob_multiplier: float = 1.0,
    ):
        """
        Perform decoding with nucleous sampling and temperature
        """
        prompt_encoded = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0).to(self.cfg.trainer.device)
        generated, eos_probs = self.model.generate(
            prompt_encoded,
            eos_token_id,
            top_p=top_p,
            temperature=temperature,
            max_steps=max_steps,
            eos_prob_multiplier=eos_prob_multiplier,
        )
        decoded = self.tokenizer.decode(generated[0].cpu().tolist())
        return decoded, eos_probs

    @property
    def current_device(self):
        if self.is_distributed:
            return self.model.module.embedding.weight.data.device
        else:
            return self.model.embedding.weight.data.device

    def validate(self):
        mem("before validation")
        self.model.eval()
        val_iters = 0
        val_loss_epoch = torch.zeros((), device=self.current_device, dtype=torch.float32)
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
                val_loss, _ = cross_entropy(logits, targets, weight=None, per_token_byte_lengths=self.id2byte_len)
                val_loss *= self.val_loss_multiplier
                val_loss_epoch += val_loss.to(torch.float32)
                val_iters += 1
        mem("after validation")
        val_loss_epoch = (val_loss_epoch / val_iters).item()

        # metrics = self.mbbp_evaluator.evaluate_with_generate(self.generate, num_samples=1, mode="full")
        # logger.info(metrics)

        return {
            "val_loss": val_loss_epoch,
            "val_perplexity": 2.71828**val_loss_epoch,
            # "mbpp_lite_pass_1": metrics["pass_at_k"]["base"]["pass@1"]
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

        iter_wd = None
        if self.cfg.optim.use_muon:
            # second optimizer is muon
            iter_lr = self._get_lr(self.cfg.optim.muon_lr, self.cfg.optim.muon_lr * self.cfg.optim.lr_min_coeff)
            if self.cfg.optim.muon_wd_min is not None:
                # just a linear warmup
                ratio = 1 - (self.cfg.trainer.max_steps - self.iteration) / self.cfg.trainer.max_steps
                iter_wd = self.cfg.optim.muon_wd_min + ratio * (self.cfg.optim.muon_wd - self.cfg.optim.muon_wd_min)

            for pg in self.optimizers[1].param_groups:
                # momentum warmup for fixed 300 steps
                momentum = self.cfg.optim.betas[0]
                pg["momentum"] = 0.85 + min(max(self.iteration, 1), 300) / 300 * (momentum - 0.85)
                pg["lr"] = iter_lr
                if iter_wd is not None:
                    pg["wd"] = iter_wd
        return iter_lr, iter_wd if iter_wd is not None else self.cfg.optim.muon_wd

    def train_step(self, inputs, targets):
        self.model.train()
        iter_lr, iter_wd = self._set_lr()

        z_loss_weight = self.cfg.trainer.z_loss_weight

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()

        with torch.autocast("cuda", enabled=self.cfg.trainer.dtype == "bfloat16"):
            logits, avg_kurtosis_values = self.model(inputs)
            # using bits-per-bytes formulation of loss
            loss, z_loss = cross_entropy(logits, targets, weight=None, per_token_byte_lengths=self.id2byte_len)
            loss *= self.train_loss_multiplier
            z_loss *= self.train_loss_multiplier
        if self.cfg.trainer.gradient_accumulation_steps > 1:
            loss /= self.cfg.trainer.gradient_accumulation_steps
            z_loss /= self.cfg.trainer.gradient_accumulation_steps

        loss_for_backward = loss + z_loss_weight * z_loss

        if self.is_distributed:
            coef = self.train_dataset.local_batch_size / self.cfg.data.batch_size
            loss_for_backward *= coef

        context = self.model.no_sync() if self.is_distributed else nullcontext()
        with context:
            if self.cfg.trainer.dtype == "bfloat16":
                self.scaler.scale(loss_for_backward).backward()
            else:
                loss_for_backward.backward()

        # TODO: this will block execution of cuda stream so remove when not needed
        end.record()
        end.synchronize()

        update_ratio = 0.0
        time_comm = 0.0
        if self.iteration % self.cfg.trainer.gradient_accumulation_steps == 0:
            # DDP
            if self.is_distributed:
                time_comm = self.model.finish_gradient_synchronization()

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
        log = {
            "train_loss": train_loss * self.cfg.trainer.gradient_accumulation_steps,
            "z_loss": z_loss * self.cfg.trainer.gradient_accumulation_steps,
            "grad_norm": grad_norm,
            "learning_rate": iter_lr,
            "update_ratio": update_ratio,
            "avg_kurtosis": avg_kurtosis_values,
            "parameter_norms": parameter_norms,
            "time_comm": time_comm,
            f"rank_{self.rank}_fwd_bwd": start.elapsed_time(end) / 1e3,
        }
        if iter_wd is not None:
            log["wd"] = iter_wd
        return log

    def postprocess_step_stats(self, step_stats):
        log_processed = {
            "train_loss": step_stats["train_loss"].item(),
            "z_loss": step_stats["z_loss"].item(),
            "grad_norm": step_stats["grad_norm"].item(),
            "avg_kurtosis": step_stats["avg_kurtosis"].cpu().tolist(),
            "parameter_norms": {k: v.item() for k, v in step_stats["parameter_norms"].items()},
        }
        # take any other key as it is
        return {**log_processed, **{k: v for k, v in step_stats.items() if k not in log_processed}}

    def train(self):
        logger.info("Starting training loop")
        while self.iteration < self.cfg.trainer.max_steps:
            if self.rank_zero_only:
                if self.iteration % self.cfg.trainer.save_every == 0:
                    self.save_state()
                if self.iteration > 0 and self.iteration % self.cfg.trainer.val_every == 0:
                    val_metrics = self.validate()
                    self.log(val_metrics)

            inputs, targets = self.train_dataset.get_batch(self.cfg.data.batch_size)
            t0 = time.monotonic()
            raw_step_stats = self.train_step(inputs, targets)
            t1 = time.monotonic()
            if self.iteration % self.cfg.trainer.log_every == 0 and self.rank_zero_only:
                logger.info(f"Train iteration {self.iteration}")
                # call item and introduce sync
                step_stats = self.postprocess_step_stats(raw_step_stats)
                kurtosis_log = {f"log/block{i}_kurtosis": v for i, v in enumerate(step_stats["avg_kurtosis"])}
                pnorm_log = {f"log/{k}_norm": v for k, v in step_stats["parameter_norms"].items()}
                self.log(
                    {
                        "train/loss": step_stats["train_loss"],
                        "train/z_loss": step_stats["z_loss"],
                        "train/grad_norm": step_stats["grad_norm"],
                        "train/learning_rate": step_stats["learning_rate"],
                        "train/wd": step_stats["wd"],
                        "train/update_ratio": step_stats["update_ratio"],
                        "train/step": self.iteration,
                        "train/step_time": t1 - t0,
                        "train/step_comm_time": step_stats["time_comm"],
                        "train/tokens_processed": self.tokens_processed,
                        **kurtosis_log,
                        **pnorm_log,
                    }
                )
                # rank fwd_bwd time on each rank
                self.log(
                    {f"train/rank_{self.rank}_fwd_bwd_time": raw_step_stats[f"rank_{self.rank}_fwd_bwd"]}, rank_zero=False
                )
            self.iteration += 1
        if self.rank_zero_only:
            self.save_state()
            val_metrics = self.validate()
            self.log(val_metrics)
