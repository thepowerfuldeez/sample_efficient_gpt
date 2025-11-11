import torch
from torch import Tensor
from jaxtyping import Float, Int


from sample_efficient_gpt.transformer.ops.cross_entropy import LigerCrossEntropyFunction
from sample_efficient_gpt.utils.profiling import nvtx_range


@nvtx_range("cross entropy")
def cross_entropy(
    logits: Float[Tensor, "... seq vocab_size"], targets: Int[Tensor, "... seq"], weight, per_token_byte_lengths=None
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Compute the cross-entropy loss for a sequence of logits and targets.
    Formula:
        loss = -sum(log(p(y_i | x_i))) / n
        where p(y_i | x_i) is the probability of the target y_i given the input x_i,
        and n is the number of tokens in the sequence.

        p(y_i | x_i) = exp(logits[y_i]) / sum(exp(logits[j])) (softmax)

    Args:
        logits: Tensor of shape (..., seq, vocab_size) containing the logits for each token in the vocabulary.
        targets: Tensor of shape (..., seq) containing the target indices for each token in the sequence.
        weight: sample weight per token, should be normalized
        per_token_byte_lengths: length of bytes per each token obtained from tokenizer, useful for bits-per-byte reporting

    Returns:
        Tensor of shape (..., seq) containing the cross-entropy loss for each token in the sequence.
    """
    m: Float[Tensor, "... seq"] = logits.max(dim=-1).values
    target_logits: Float[Tensor, "... seq"] = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    with torch.autocast("cuda", enabled=False):
        # lg: [seq, vocab_size]; m: [bs, seq]
        logsumexp_values = torch.stack(
            [torch.logsumexp(lg - m[i].unsqueeze(-1), dim=-1) for i, lg in enumerate(logits)]
        )
        loss: Float[Tensor, "... seq"] = m - target_logits + logsumexp_values

    if weight is not None:
        pass
    if per_token_byte_lengths is not None:
        byte_len = per_token_byte_lengths[targets.view(-1)].view_as(targets)  # bytes/token
        nats_per_byte = (loss.detach() * byte_len).sum() / byte_len.sum()
        loss_bpb = nats_per_byte * 1.44269  # 1 / log(2)
    else:
        loss_bpb = None
    loss = loss.mean()
    z_loss = (logsumexp_values**2).mean()
    return loss, z_loss, loss_bpb


def cross_entropy_(
    logits: Float[Tensor, "... seq vocab_size"],
    targets: Int[Tensor, "... seq"],
):
    m: Float[Tensor, "... seq"] = logits.max(dim=-1).values
    target_logits: Float[Tensor, "... seq"] = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    with torch.autocast("cuda", enabled=False):
        # lg: [seq, vocab_size]; m: [bs, seq]
        logsumexp_values = torch.stack(
            [torch.logsumexp(lg - m[i].unsqueeze(-1), dim=-1) for i, lg in enumerate(logits)]
        )
        loss: Float[Tensor, "... seq"] = m - target_logits + logsumexp_values
    return loss, (logsumexp_values**2)


@nvtx_range("cross entropy")
def efficient_cross_entropy(
    logits: Float[Tensor, "... seq vocab_size"],
    targets: Int[Tensor, "... seq"],
    weight=None,
    per_token_byte_lengths=None,
) -> tuple[Tensor, Tensor, Tensor]:
    # by not reducing loss here we would introduce extra graph node,
    # but it's the easiest way to include BPB without modifying the kernel
    loss, z_loss = LigerCrossEntropyFunction.apply(
        logits.view(-1, logits.size(-1)),
        targets.view(-1),
        None,
        -100,
        1e-6,
        0.0,
        "none",
        None,
        True,
    )
    if weight is not None:
        pass
    if per_token_byte_lengths is not None:
        byte_len = per_token_byte_lengths[targets.view(-1)].view_as(targets)  # bytes/token
        nats_per_byte = (loss.detach().view_as(targets) * byte_len).sum() / byte_len.sum()
        loss_bpb = nats_per_byte * 1.44269  # 1 / log(2)
    else:
        loss_bpb = None
    loss = loss.mean()
    z_loss = z_loss.mean()
    return loss, z_loss, loss_bpb
