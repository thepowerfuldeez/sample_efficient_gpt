import torch
from torch import Tensor
from jaxtyping import Float, Int


# @nvtx.range("cross entropy")
def cross_entropy(logits: Float[Tensor, "... seq vocab_size"], targets: Int[Tensor, "... seq"]) -> Tensor:
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
    return loss.mean(), (logsumexp_values**2).mean()
