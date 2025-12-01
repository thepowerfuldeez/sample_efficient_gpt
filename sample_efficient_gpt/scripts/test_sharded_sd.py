import torch
import torch.nn as nn
import torch.distributed as dist

from sample_efficient_gpt.training.distributed import ShardedOptimizer

dist.init_process_group("gloo")

m = nn.Sequential(
    nn.Linear(100, 100),
    nn.Linear(100, 200),
    nn.Linear(200, 300),
    nn.Linear(300, 100),
)

r = dist.get_rank()
opt = ShardedOptimizer(m.parameters(), torch.optim.AdamW, lr=1e-3)

for _ in range(5):
    opt.zero_grad()
    x = torch.randn(1, 100)
    m(x).sum().backward()
    opt.step()

# print(r, opt.state_dict())
full_sd = opt.state_dict()
print("full sd", full_sd['param_groups'])
opt.load_state_dict(full_sd)
# print(r, opt.state_dict())