import os
from contextlib import nullcontext, contextmanager

import torch

# Only use NVTX when env var is provided
ENABLE_PROF = os.getenv("ENABLE_PROF", "0") == "1"

if ENABLE_PROF:
    nvtx_range = torch.cuda.nvtx.range
else:
    # no-op that ALSO works as decorator and context manager
    @contextmanager
    def nvtx_range(_name=None):
        yield
