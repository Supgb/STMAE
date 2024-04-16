import torch
import torch.nn as nn

from collections import defaultdict


def dice_loss(input_mask, cls_gt):
    num_objects = input_mask.shape[1]
    losses = []
    for i in range(num_objects):
        mask = input_mask[:,i].flatten(start_dim=1)
        # background not in mask, so we add one to cls_gt
        gt = (cls_gt==(i+1)).float().flatten(start_dim=1)
        numerator = 2 * (mask * gt).sum(-1)
        denominator = mask.sum(-1) + gt.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        losses.append(loss)
    return torch.cat(losses).mean()

class LossComputer:
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.ce = nn.CrossEntropyLoss()

    def compute(self, data, b, t, num_objects):
        losses = defaultdict(int)
        total_loss = 0
        for bi in range(b):
            loss_t = self.ce(data[f'logits_{t-1}'][bi:bi+1, :num_objects[bi]+1], data['cls_gt'][bi:bi+1,0,0])
            losses[f'recon_loss_{t-1}'] += loss_t / b
        total_loss += losses[f'recon_loss_{t-1}']

        losses[f'dice_loss_{t-1}'] = dice_loss(data[f'masks_{t-1}'], data['cls_gt'][:,0,0])
        total_loss += losses[f'dice_loss_{t-1}']
        
        losses['total_loss'] = total_loss

        return losses
