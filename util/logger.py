"""
Dumps things to tensorboard and console
"""

import os
import warnings

import torch.distributed as dist
import torchvision.transforms as transforms
import neptune
from neptune.types import File
import time


def tensor_to_numpy(image):
    image_np = (image.numpy() * 255).astype('uint8')
    return image_np

def detach_to_cpu(x):
    return x.detach().cpu()

def fix_width_trunc(x):
    return ('{:.9s}'.format('{:0.9f}'.format(x)))

class NeptuneLogger:
    def __init__(self, short_id, id, git_info, custom_run_id, monitoring_namespace):
        self.short_id = short_id
        if self.short_id == 'NULL':
            self.short_id = 'DEBUG'

        if id is None:
            self.no_log = True
            self.logger = {}
            warnings.warn('Logging has been disbaled.')
        else:
            self.no_log = False

            self.inv_im_trans = transforms.Normalize(
                mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
                std=[1/0.229, 1/0.224, 1/0.225])

            self.inv_seg_trans = transforms.Normalize(
                mean=[-0.5/0.5],
                std=[1/0.5])

            self.logger = neptune.init_run(
                source_files=["**/*.py", "scripts/*.sh"],
                custom_run_id=custom_run_id,
                monitoring_namespace=monitoring_namespace,
                project=os.environ['NEPTUNE_PROJ_NAME'],
                api_token=os.environ['NEPTUNE_TOKEN'],
            )

        self.log_string('git', git_info)

    def log_scalar(self, tag, x, step=None):
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        if dist.get_rank() == 0:
            self.logger[tag].append(x, step=step)

    def log_metrics(self, l1_tag, l2_tag, val, step=None, f=None, log_neptune=True):
        tag = l1_tag + '/' + l2_tag
        text = '{:s} - It {:6d} [{:5s}] [{:13}]: {:s}'.format(self.short_id, step, l1_tag.upper(), l2_tag, fix_width_trunc(val))
        print(text)
        if not log_neptune:
            return
        if f is not None:
            f.write(text + '\n')
            f.flush()
        self.log_scalar(tag, val, step)
        time.sleep(0.1)  # TODO: This is a hack for neptune to catch up. Remove this when neptune is fixed.

    def log_im(self, tag, x, step=None):
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        if dist.get_rank() == 0:
            x = detach_to_cpu(x)
            x = self.inv_im_trans(x)
            x = tensor_to_numpy(x)
            self.logger[tag].append(File.as_image(x), step=step)

    def log_cv2(self, tag, x, step=None):
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        # x = x.transpose((2, 0, 1))
        if dist.get_rank() == 0:
            self.logger[tag].append(File.as_image(x), step=step)

    def log_seg(self, tag, x, step=None):
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        if dist.get_rank() == 0:
            x = detach_to_cpu(x)
            x = self.inv_seg_trans(x)
            x = tensor_to_numpy(x)
            self.logger[tag].append(File.as_image(x), step=step)

    def log_gray(self, tag, x, step=None):
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        if dist.get_rank() == 0:
            x = detach_to_cpu(x)
            x = tensor_to_numpy(x)
            self.logger[tag].append(File.as_image(x), step=step)

    def log_string(self, tag, x):
        print(tag, x)
        if self.no_log:
            warnings.warn('Logging has been disabled.')
            return
        if dist.get_rank() == 0:
            self.logger[tag] = x
