import math
import torch
from inference.kv_memory_store import KeyValueMemoryStore
from model.memory_util import get_similarity, do_softmax


def softmax_w_top(x, top):
    values, indices = torch.topk(x, k=top, dim=1)
    x_exp = torch.exp(values - values[:,0])

    x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
    x_exp /= x_exp_sum
    x.zero_().scatter_(1, indices, x_exp) # B * THW * HW

    return x


class MemoryBank:
    def __init__(self, top_k):
        self.top_k = top_k

        self.CK = None
        self.CV = None

        self.mem = KeyValueMemoryStore(False)

    def _global_matching(self, mk, qk):
        # NE means number of elements -- typically T*H*W
        B, CK, NE = mk.shape

        # a = mk.pow(2).sum(1).unsqueeze(2)
        # b = 2 * (mk.transpose(1, 2) @ qk)
        # c = qk.pow(2).expand(B, -1, -1).sum(1).unsqueeze(1)

        # affinity = (-a+b-c) / math.sqrt(CK)  # B, NE, HW
        affinity = ('bcn,bcN->bnN', mk, qk)
        affinity = softmax_w_top(affinity, top=self.top_k)  # B, THW, HW

        return affinity

    def _readout(self, affinity, mv):
        return mv @ affinity

    def match_memory(self, qk):
        num_groups = self.mem.num_groups
        _, _, h, w = qk.shape

        qk = qk.flatten(start_dim=2)
        
        similarity = get_similarity(self.mem.key, qk)
        affinity = do_softmax(similarity, top_k=self.top_k, inplace=num_groups==1)

        affinity = [affinity]

        # compute affinity group by group as later groups only have a subset of keys
        for gi in range(1, num_groups):
            affinity_one_group = do_softmax(similarity[:, -self.mem.get_v_size(gi):], 
                top_k=self.top_k, inplace=(gi==num_groups-1))
            affinity.append(affinity_one_group)
            
        all_memory_value = self.mem.value

        # Shared affinity within each group
        all_readout_mem = torch.cat([
            self._readout(affinity[gi], gv)
            for gi, gv in enumerate(all_memory_value)
        ], 0)

        return all_readout_mem.view(self.CV, all_readout_mem.shape[0], h, w)

    def add_memory(self, key, value, objects):
        self.CK = key.shape[1]
        self.CV = value.shape[1]

        key = key.flatten(start_dim=2)
        value = value.flatten(start_dim=2)
        self.mem.add(key, value, objects)