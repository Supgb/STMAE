"""
trainer.py - warpper and utility functions for network training
Compute loss, back-prop, update parameters, logging, etc.
"""


import os
import time
import copy

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from model.network import STMAE
from model.losses import LossComputer
from util.log_integrator import Integrator
from util.image_saver import pool_pairs
from util.davis2017 import evaluate_semi
from util.tensor_util import pad_divide_by, unpad


class STMAETrainer:
    def __init__(self, config, logger=None, save_path=None, local_rank=0, world_size=1):
        self.config = config
        self.num_frames = config['num_frames']
        self.num_ref_frames = config['num_ref_frames']
        self.local_rank = local_rank

        self.STMAE = nn.parallel.DistributedDataParallel(
            STMAE(config).cuda(), 
            device_ids=[local_rank], output_device=local_rank, broadcast_buffers=False)

        # Set up logger when local_rank=0
        self.logger = logger
        self.save_path = save_path if local_rank == 0 else None
        if logger is not None:
            self.last_time = time.time()
            self.logger.logger['model_size'] = str(sum([param.nelement() for param in self.STMAE.parameters()]))
        self.val_integrator = Integrator(self.logger, distributed=False, local_rank=local_rank, world_size=world_size)
        self.train_integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=world_size)
        self.loss_computer = LossComputer(config)

        self.train()
        self.optimizer = optim.AdamW(filter(
            lambda p: p.requires_grad, self.STMAE.parameters()), lr=config['lr'], weight_decay=config['weight_decay'])
        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, config['steps'], config['gamma'])
        if config['amp']:
            self.scaler = torch.cuda.amp.GradScaler()

        # Logging info
        self.log_text_interval = config['log_text_interval']
        self.log_image_interval = config['log_image_interval']
        self.save_network_interval = config['save_network_interval']
        self.save_checkpoint_interval = config['save_checkpoint_interval']
        if config['debug']:
            self.log_text_interval = self.log_image_interval = 1

    def train_one_step(self, net, frames, gt, first_frame_gt, num_objects, num_filled_objects, selector):
        out = {}
        b, t = frames.shape[:2]
        out['cls_gt'] = gt
        out['first_frame_gt'] = first_frame_gt

        with torch.cuda.amp.autocast(enabled=self.config['amp']):
            f16, f8, f4 = net('encode_key', frames)
            f16_key = f16.transpose(1, 2)

            placeholder = torch.zeros(b, num_objects, *frames.shape[-2:], device=frames.device)
            out['pseudo_mask_0'] = first_frame_gt
            gts = F.interpolate(first_frame_gt, size=f16_key.shape[-2:], mode='bilinear', align_corners=False).unsqueeze(2)  # Bx(M+1)x1xHxW

            for ti in range(1, t):
                # Segment frame ti
                ref_gts_keys = f16_key[:, :, :ti]  # BxCxTxHxW
                ref_gts = gts[:, :, :ti]
                with torch.no_grad():
                    net.requires_grad_(False) # tricks to make ddp happy.
                    pseudo_mask_ti = net('pseudo_mask', f16_key[:, :, ti], ref_gts_keys, ref_gts)
                    pseudo_mask_ti = F.interpolate(pseudo_mask_ti, size=frames.shape[-2:], mode='bilinear', align_corners=False)
                    gt_v16 = net('encode_value', frames[:,ti], [f16[:,ti]], pseudo_mask_ti)
                    _, masks, _ = net('segment', (f16[:,ti], f8[:,ti], f4[:,ti]), gt_v16, selector)

                    new_gts = F.interpolate(masks, size=gts.shape[-2:], mode='bilinear', align_corners=False).unsqueeze(2)
                    gts = torch.cat([gts, new_gts], 2)
                    net.requires_grad_(True)

                if ti == t - 1:
                    if ti <= self.num_ref_frames:
                        ref_gts_keys = f16_key[:, :, 1:ti+1]  # BxCxTxHxW
                        ref_gts = gts[:, :, 1:ti+1]  # Bx(M+1)xTxHxW
                    else:
                        last_one = torch.ones(1, dtype=torch.int64) * (-1)
                        indices = [
                                torch.cat([torch.randperm(ti-1)[:self.num_ref_frames-1]+1, last_one])
                        for _ in range(b)]
                        ref_gts_keys = torch.stack([
                            f16_key[bi, :, indices[bi]] for bi in range(b)
                        ], 0)
                        ref_gts = torch.stack([
                            gts[bi, :, indices[bi]] for bi in range(b)
                        ], 0)
                        
                    t_pseudo_mask_ti = net('pseudo_mask', f16_key[:, :, 0], ref_gts_keys, ref_gts)
                    t_pseudo_mask_ti = F.interpolate(t_pseudo_mask_ti, size=frames.shape[-2:], mode='bilinear', align_corners=False)

                    t_gt_v16 = net('encode_value', frames[:,0], [f16[:,0]], t_pseudo_mask_ti)  # BxMxCxHxW
                    t_logits, t_masks, _ = net('segment', (f16[:,0], f8[:,0], f4[:,0]), t_gt_v16, selector)
                    out[f'logits_{ti}'] = t_logits
                else:
                    t_pseudo_mask_ti = placeholder
                    t_masks = placeholder

                out[f'f0_pseudo_masks_{ti}'] = t_pseudo_mask_ti
                out[f'query_pseudo_masks_{ti}'] = pseudo_mask_ti
                out[f'masks_{ti}'] = t_masks
                out[f'query_masks_{ti}'] = masks
            
            out['logits_0'] = first_frame_gt
            out['masks_0'] = first_frame_gt
            out['query_masks_0'] = first_frame_gt
            out['f0_pseudo_masks_0'] = first_frame_gt
            out['query_pseudo_masks_0'] = first_frame_gt

            losses = self.loss_computer.compute(out, b, t, num_filled_objects)

        return losses, out
        
    @torch.no_grad()
    def val_one_step(self, net, frames, gt, first_frame_gt):
        out = {}
        b, t = frames.shape[:2]
        frames, pad = pad_divide_by(frames, 16)
        first_frame_gt, _ = pad_divide_by(first_frame_gt, 16)
        with torch.cuda.amp.autocast(enabled=self.config['amp']):
            f16, f8, f4 = net('encode_key', frames)
            f16_key = f16.transpose(1, 2)
            
            gts = F.interpolate(first_frame_gt, size=f16_key.shape[-2:], mode='bilinear', align_corners=False).unsqueeze(2)  # Bx(M+1)x1xHxW

            for ti in range(1, t):
                ref_gts_keys = f16_key[:, :, :ti:self.config['val_mem_frames']]
                ref_gts = gts  # Bx(M+1)xTxHxW

                pseudo_mask_ti = net('pseudo_mask', f16_key[:, :, ti], ref_gts_keys, ref_gts)
                pseudo_mask_ti = F.interpolate(pseudo_mask_ti, size=frames.shape[-2:], mode='bilinear', align_corners=False)                
                gt_v16 = net('encode_value', frames[:,ti], [f16[:,ti]], pseudo_mask_ti)
                _, masks, masks_bg = net('segment', (f16[:,ti], f8[:,ti], f4[:,ti]), gt_v16, selector=None)

                out[f'f0_pseudo_masks_{ti}'] = pseudo_mask_ti
                out[f'masks_{ti}'] = masks
                out[f'masks_bg_{ti}'] = masks_bg
                out[f'masks_{ti}_pred_idx'] = masks_bg

                if ti % self.config['val_mem_frames'] == 0:
                    new_gts = F.interpolate(masks, size=gts.shape[-2:], mode='bilinear', align_corners=False).unsqueeze(2)
                    gts = torch.cat([gts, new_gts], 2)

            out['f0_pseudo_masks_0'] = first_frame_gt
            out['masks_pred_idx'] = torch.stack([out[f'masks_{ti}_pred_idx'] for ti in range(1, t)], 1).argmax(2)
            out['masks_pred_idx'] = torch.cat([gt[:, :1, 0], unpad(out['masks_pred_idx'][0], pad).unsqueeze(0)], 1)
            out['pseudo_first_frame_gt'] = first_frame_gt.unsqueeze(1)

        return out


    def do_pass(self, data, it=0, epoch=0):
        if self._is_train:
            net = self.STMAE
        else:
            net = self.STMAE.module
        # No need to store the gradient outside training
        torch.set_grad_enabled(self._is_train)

        for k, v in data.items():
            if type(v) != list and type(v) != dict and type(v) != int:
                data[k] = v.cuda(non_blocking=True)

        frames = data['rgb']
        first_frame_gt = data['first_frame_gt'][:, 0].float()
        num_objects = data['first_frame_gt'].shape[2]
        num_filled_objects = [o.item() for o in data['info']['num_objects']]
        selector = data['selector'].unsqueeze(2).unsqueeze(2)

        if self._is_train:
            losses, out = self.train_one_step(net, frames, data['cls_gt'], first_frame_gt, num_objects, num_filled_objects, selector)
            out['losses'] = losses
            out['num_filled_objects'] = num_filled_objects
            # Backward pass
            self.optimizer.zero_grad(set_to_none=True)
            if self.config['amp']:
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                losses['total_loss'].backward() 
                self.optimizer.step()
        else:
            out = self.val_one_step(net, frames, data['cls_gt'], first_frame_gt)
        
            # Logging
            if self._do_log:
                if self.logger is not None:
                    images = {**data, **out}
                    size = (384, 384)
                    self.logger.log_cv2(f'val/pairs_{it}', pool_pairs(images, size, num_filled_objects, step=5, mode='val'), epoch)

        return {**data, **out}

    @torch.no_grad()
    def do_eval(self, epoch, val_loader):
        def davis_masks(masks, num_objects):
            masks = masks.cpu()
            tmp = torch.ones(num_objects, *masks.shape)
            tmp = tmp * torch.arange(1, num_objects + 1)[:, None, None, None]
            return (tmp == masks[None, ...]).long().numpy()
        
        Js = {"M": [], "R": [], "D": []}
        Fs = {"M": [], "R": [], "D": []}

        # for single-object pretraining.
        value_encoder = self.STMAE.module.value_encoder
        single_object = value_encoder.single_object
        value_encoder.single_object = False
        if single_object:
            original_weights = copy.deepcopy(value_encoder.conv1.weight.data)
            pads = torch.zeros((64, 1, 7, 7), device=value_encoder.conv1.weight.device)
            value_encoder.conv1.weight.data = torch.cat([value_encoder.conv1.weight, pads], 1)

        print("Start validation...")
        for it, batch in enumerate(val_loader):  # batch size = 1.
            seq_name = batch['info']['name'][0]
            print(f"Sequence {seq_name}:")
            out = self.do_pass(batch, it, epoch)

            num_objects = int(out["cls_gt"].max())
            masks_gt = davis_masks(out["cls_gt"].squeeze(), num_objects)
            masks_pred = davis_masks(out["masks_pred_idx"].squeeze(), num_objects)
            assert masks_gt.shape == masks_pred.shape

            metrics = evaluate_semi((masks_gt,), (masks_pred,))
            J, F = metrics['J'], metrics['F']

            print("Jaccard: ", J["M"])
            print("F-score: ", F["M"])

            for i in ("M", "R", "D"):
                Js[i] += J[i]
                Fs[i] += F[i]

        mJ_F = (np.mean(Js["M"]) + np.mean(Fs["M"])) * 0.5
        str_metrics = ['J&F-mean', 'J-mean', 'J-recall', 'J-decay', 'F-mean', 'F-recall', 'F-decay']
        res_metrics = [mJ_F, np.mean(Js["M"]), np.mean(Js["R"]), np.mean(Js["D"]), np.mean(Fs["M"]), np.mean(Fs["R"]), np.mean(Fs["D"])]
        for (name, val) in zip(str_metrics, res_metrics):
            self.logger.log_metrics('val', name, val, epoch)

        if single_object:
            value_encoder.single_object = single_object
            value_encoder.conv1.weight.data = original_weights

        return mJ_F

    def save_network(self, it, epoch, jf=0.):
        if self.save_path is None:
            print('Saving has been disabled.')
            return
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = f'{self.save_path}_epoch_{epoch}_{it}_jf_{jf:.2f}.pth'
        torch.save(self.STMAE.module.state_dict(), model_path)
        print(f'Network saved to {model_path}.')

    def save_checkpoint(self, it, epoch):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = f'{self.save_path}_epoch_{epoch}_checkpoint_{it}.pth'
        checkpoint = {
            'epoch': epoch,
            'it': it,
            'network': self.STMAE.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)
        print(f'Checkpoint saved to {checkpoint_path}.')

    def load_checkpoint(self, path):
        # This method loads everything and should be used to resume training
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        map_location = 'cuda:%d' % self.local_rank
        self.STMAE.module.load_state_dict(network)
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Network weights, optimizer states, and scheduler states loaded.')

        return it

    def load_network_in_memory(self, src_dict):
        self.STMAE.module.load_weights(src_dict)
        print('Network weight loaded from memory.')

    def load_network(self, path):
        # This method loads only the network weight and should be used to load a pretrained model
        map_location = 'cuda:%d' % self.local_rank
        try:
            src_dict = torch.load(path, map_location={'cuda:0': map_location})['network']
        except:
            src_dict = torch.load(path, map_location={'cuda:0': map_location})

        self.load_network_in_memory(src_dict)
        print(f'Network weight loaded from {path}')

    def train(self):
        self._is_train = True
        self._do_log = True
        self.integrator = self.train_integrator
        self.STMAE.eval()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.integrator = self.val_integrator
        self.STMAE.eval()
        return self

    def test(self):
        self._is_train = False
        self._do_log = False
        self.STMAE.eval()
        return self

