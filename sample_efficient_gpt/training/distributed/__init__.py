from sample_efficient_gpt.training.distributed.ddp_bucket import DDP
# from functools import partial
# from torch.nn.parallel import DistributedDataParallel as DDP
# from torch.nn.parallel.distributed import _MixedPrecision
# import torch

# DDP = partial(
#     DDP,
#     static_graph=True,
#     broadcast_buffers=False,
#     gradient_as_bucket_view=True,
#     bucket_cap_mb=32,
#     mixed_precision=_MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
# )





__all__ = ["DDP"]
