import torch
import torch.distributed as dist
from torch.optim import Optimizer


class ShardedOptimizer(Optimizer):
    def __init__(self, params, optimizer_cls: type[Optimizer], **kwargs):
        self.sharded_params = []
        params = list(params)
        super().__init__(params, kwargs)

        assert dist.is_initialized()
        self.rank, self.world_size = dist.get_rank(), dist.get_world_size()
        assert not isinstance(params[0], dict), "param groups are not supported"
        n = len(params)
        
        # determine chunks to cover all params
        chunk_size = n // self.world_size
        indices = [0]
        for i in range(self.world_size):
            new = min(indices[-1] + chunk_size, n)
            if i == self.world_size - 1:
                new += (n - new)
            indices.append(new)
        self.params_of_rank = [params[l:r] for l, r in zip(indices[:-1], indices[1:])]
        self.sharded_params = self.params_of_rank[self.rank]

        self.optimizer = optimizer_cls(self.sharded_params, **kwargs)
        self.params = params

        print(f"params of rank {self.rank} is {[x.shape for x in self.params_of_rank[self.rank]]}")

        # print(f"[rank={self.rank}] {len(self.sharded_params)=} {len(params)=}")

    def zero_grad(self, **kwargs):
        for p in self.params:
            p.grad = None

    def step(self, closure=None, **kwargs):
        retval = self.optimizer.step(closure=closure, **kwargs)
        # print(f"[rank={self.rank}] step done")

        handles = []
        for src_rank in range(self.world_size):
            for p in self.params_of_rank[src_rank]:
                # we are the sender
                if src_rank == self.rank:
                    buf = p.detach().clone()
                # we are now the receiver
                else:
                    buf = torch.empty_like(p)
                
                handle = dist.broadcast(buf, src=src_rank, async_op=True)
                def _finish(_, buf=buf, p=p, src_rank=src_rank):
                    # copy weight after receiving
                    if src_rank != self.rank:
                        with torch.no_grad():
                            p.copy_(buf)
                handle.get_future().then(_finish)
                handles.append(handle)
        for h in handles:
            h.wait()
        return retval
    
    def state_dict(self):
        offset = 0
        if self.rank > 0:
            offset = len(self.params_of_rank[self.rank - 1])
        sd = self.optimizer.state_dict()

        param_groups = []
        for pg in sd['param_groups']:
            pg_rank = {**pg, "params": [p + offset for p in pg['params']]}
            print("pg rank", pg_rank)
            if pg_rank['params']:
                param_groups.append(pg_rank)

        new_state = {offset + i: v for i, v in sd['state'].items()}
        new_sd = {"param_groups": param_groups, "state": new_state}
        gathered_sd = [None for _ in range(self.world_size)]
        dist.all_gather_object(gathered_sd, new_sd)
        combined_sd = {"state": {}, "param_groups": []}
        for sd in gathered_sd:
            combined_sd['state'] = {**sd['state'], **combined_sd['state']}
            combined_sd['param_groups'].extend(sd['param_groups'])
        return combined_sd
    
    def load_state_dict(self, state_dict):
        offset = 0
        if self.rank > 0:
            offset = len(self.params_of_rank[self.rank - 1])
        count = len(self.params_of_rank[self.rank])

        param_groups = []
        for pg in state_dict['param_groups']:
            pg_rank = {**pg, "params": [p - offset for p in pg['params'] if p >= offset and p < offset + count]}
            if pg_rank['params']:
                param_groups.append(pg_rank)
        sd_for_rank = {"param_groups": param_groups}
        sd_for_rank['state'] = {k - offset: v for k, v in state_dict['state'].items() if k >= offset and k < offset + count}
        print("sd for rank", self.rank, sd_for_rank['state'].keys(), sd_for_rank['param_groups'])
        self.optimizer.load_state_dict(sd_for_rank)