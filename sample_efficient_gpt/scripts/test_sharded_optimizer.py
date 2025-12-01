import torch
import torch.nn as nn
from copy import deepcopy
import torch.distributed as dist
import numpy as np

from sample_efficient_gpt.transformer.core import Linear
from sample_efficient_gpt.training.distributed import ZeRO2


torch.manual_seed(42)

if __name__ == "__main__":
    dist.init_process_group("gloo")
    model = nn.Sequential(*[Linear(2, 2) for _ in range(5)])
    model = model

    model2 = deepcopy(model)
    model = ZeRO2(model, torch.optim.AdamW, set([n for n, _ in model.named_parameters()]), {"lr": 10})

    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    rank = dist.get_rank()
    if rank == 0:
        print(f"{rank=} model {[p.data for p in model.parameters()]}")
        print()    
        print()    
        print()    
        print()    


    # # sharded_opt = Zero2ShardedOptimizer(model.parameters(), torch.optim.AdamW, lr=10)
    # sharded_opt.zero_grad()
    model.zero_grad()

    opt = torch.optim.AdamW(model2.parameters(), lr=10)
    opt.zero_grad()

    for _ in range(3):
        x = torch.randn(1, 2)
        model(x).sum().backward()
        model.finish_gradient_synchronization()
        model2(x).sum().backward()
        
        model.step()
        opt.step()

    print(f"{rank=} sharded model {[p.data for p in model.parameters()]}")
    print("-----------")

    print(f"{rank=} model {[p.data for p in model2.parameters()]}")
    print("-----------")

    for non_sharded_parameters, sharded_parameters in zip(model2.parameters(), model.parameters()):
        np.testing.assert_allclose(
            non_sharded_parameters.detach().cpu().numpy(),
            sharded_parameters.detach().cpu().numpy(),
        )

    if rank == 0:
        print(model.optimizer.state_dict())
        sd = opt.state_dict()
        print("sd", sd)