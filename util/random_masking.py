"""
Copied from MAE.
"""
import torch

def patchify(imgs, p):
    """
    imgs: (N, (M+1), H, W)
    x: (N, L, patch_size**2, (M+1))
    """
    assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

    h = w = imgs.shape[2] // p
    x = imgs.reshape(shape=(*imgs.shape[:2], h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p**2, imgs.shape[1]))
    return x

def unpatchify(x, p):
    """
    x: (N, L, patch_size**2, (M+1))
    imgs: (N, (M+1), H, W)
    """
    h = w = int(x.shape[1]**.5)
    assert h * w == x.shape[1]
    
    x = x.reshape(shape=(x.shape[0], h, w, p, p, -1))
    x = torch.einsum('nhwpqc->nchpwq', x)
    imgs = x.reshape(shape=(x.shape[0], -1, h * p, h * p))
    return imgs

def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D, M = x.shape  # batch, length, dim
    len_keep = int(L * (1 - mask_ratio))
    
    noise = torch.rand(N, L, M, device=x.device)  # noise in [0, 1]
    
    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
    # ids_restore = torch.argsort(ids_shuffle, dim=1)

    # keep the first subset
    mask = torch.zeros_like(x, dtype=torch.bool)
    # ids_keep = ids_shuffle[:, :len_keep]
    ids_drop = ids_shuffle[:, len_keep:].unsqueeze(2).expand(-1, -1, D, -1)
    x_masked = x.clone()
    x_masked.scatter_(1, ids_drop, 0)
    mask.scatter_(1, ids_drop, 1)

    # generate the binary mask: 0 is keep, 1 is remove
    # mask = torch.ones([N, L], device=x.device)
    # mask[:, :len_keep] = 0
    # # unshuffle to get the binary mask
    # mask = torch.gather(mask, dim=1, index=ids_restore)

    return x_masked, mask

# def random_masking(x, mask_ratio):
#     N, L, D, M = x.shape

#     # Create a random tensor with the same shape as mask
#     mask = torch.bernoulli(torch.rand(N, L, 1, M, device=x.device) * (1-mask_ratio)).to(torch.bool).expand(-1, -1, D, -1)
    
#     # Multiply the mask with the random tensor
#     x_masked = x * mask
    
#     return x_masked, ~mask