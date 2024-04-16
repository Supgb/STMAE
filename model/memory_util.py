import math
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional


def get_similarity(mk, qk):
    # used for training/inference and memory reading/memory potentiation
    # mk: B x CK x [N]    - Memory keys
    # ms: B x  1 x [N]    - Memory shrinkage
    # qk: B x CK x [HW/P] - Query keys
    # qe: B x CK x [HW/P] - Query selection
    # Dimensions in [] are flattened
    CK = mk.shape[1]
    mk = mk.flatten(start_dim=2)
    qk = qk.flatten(start_dim=2)

    # similar to STCN if we don't have the selection term
    similarity = torch.einsum('bcn,bcN->bnN', mk, qk)

    return similarity

def do_softmax(similarity, top_k: Optional[int]=None, inplace=False, return_usage=False):
    # normalize similarity with top-k softmax
    # similarity: B x N x [HW/P]
    # use inplace with care
    if top_k is not None:
        values, indices = torch.topk(similarity, k=top_k, dim=1)

        x_exp = values.exp_()
        x_exp /= torch.sum(x_exp, dim=1, keepdim=True)
        if inplace:
            similarity.zero_().scatter_(1, indices, x_exp) # B*N*HW
            affinity = similarity
        else:
            affinity = torch.zeros_like(similarity).scatter_(1, indices, x_exp) # B*N*HW
    else:
        maxes = torch.max(similarity, dim=1, keepdim=True)[0]
        x_exp = torch.exp(similarity - maxes)
        x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
        affinity = x_exp / x_exp_sum 
        indices = None

    if return_usage:
        return affinity, affinity.sum(dim=2)

    return affinity

def get_affinity(mk, qk, ms=None, qe=None):
    # shorthand used in training with no top-k
    similarity = get_similarity(mk, qk, ms, qe)
    affinity = do_softmax(similarity)
    return affinity

def readout(affinity, mv):
    B, CV, T, H, W = mv.shape

    mo = mv.view(B, CV, T*H*W) 
    mem = torch.bmm(mo, affinity)
    mem = mem.view(B, CV, H, W)

    return mem

def get_masks_affinity(mk, qk, top_k=None):  # mk: BxCxTxHxW qk: BxCxHxW
    mk = mk.flatten(2)
    qk = qk.flatten(2)
    similarity = torch.einsum('bcn,bcN->bnN', mk, qk)  # B x THW x HW
    affinity = do_softmax(similarity, top_k=top_k)
    # affinity = similarity.softmax(dim=1)
    return affinity

def readout_masks(affinity, mv):  # BxCxTxHxW
    b, c, t, h, w = mv.shape
    mv = mv.reshape(b,c,t*h*w)
    masks = torch.bmm(mv, affinity)  # BxCxHW
    return masks.reshape(b, c, h, w)