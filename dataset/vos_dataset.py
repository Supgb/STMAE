import os
from os import path, replace

import torch
from torch.utils.data.dataset import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from PIL import Image
import numpy as np

from dataset.range_transform import im_normalization, im_mean
from dataset.reseed import reseed

from util.palette import custom_palette


class DLBase(Dataset):

    def __init__(self, *args, **kwargs):
        super(DLBase, self).__init__(*args, **kwargs)

        # RGB
        self.MEAN = [0.485, 0.456, 0.406]
        self.STD = [0.229, 0.224, 0.225]

        self._init_means()

    def _init_means(self):
        self.MEAN255 = [255.*x for x in self.MEAN]
        self.STD255 = [255.*x for x in self.STD]

    def _init_palette(self, num_classes):
        self.palette = custom_palette(num_classes)

    def get_palette(self):
        return self.palette

    def remove_labels(self, mask):
        # Remove labels not in training
        for ignore_label in self.ignore_labels:
            mask[mask == ignore_label] = 255

        return mask.long()


class VOSDataset(Dataset):
    """
    Works for DAVIS/YouTubeVOS/BL30K training
    For each sequence:
    - Pick three frames
    - Pick two objects
    - Apply some random transforms that are the same for all frames
    - Apply random transform to each of the frame
    - The distance between frames is controlled
    """
    def __init__(self, im_root, gt_root, max_jump, is_bl, subset=None, num_frames=3, max_num_obj=3, finetune=False):
        self.im_root = im_root
        self.gt_root = gt_root
        self.max_jump = max_jump
        self.is_bl = is_bl
        self.num_frames = num_frames
        self.max_num_obj = max_num_obj

        self.videos = []
        self.frames = {}

        vid_list = sorted(os.listdir(self.im_root))
        # Pre-filtering
        for vid in vid_list:
            if subset is not None:
                if vid not in subset:
                    continue
            frames = sorted(os.listdir(os.path.join(self.im_root, vid)))
            if len(frames) < num_frames:
                continue
            self.frames[vid] = frames
            self.videos.append(vid)

        print('%d out of %d videos accepted in %s.' % (len(self.videos), len(vid_list), im_root))

        # These set of transform is the same for im/gt pairs, but different among the 3 sampled frames
        self.pair_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.01, 0.01, 0.01, 0),
        ])

        self.pair_im_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if finetune or self.is_bl else 15, shear=0 if finetune or self.is_bl else 10, interpolation=InterpolationMode.BILINEAR, fill=im_mean),
        ])

        self.pair_gt_dual_transform = transforms.Compose([
            transforms.RandomAffine(degrees=0 if finetune or self.is_bl else 15, shear=0 if finetune or self.is_bl else 10, interpolation=InterpolationMode.NEAREST, fill=0),
        ])

        # These transform are the same for all pairs in the sampled sequence
        self.all_im_lone_transform = transforms.Compose([
            transforms.ColorJitter(0.1, 0.03, 0.03, 0),
            transforms.RandomGrayscale(0.05),
        ])

        if self.is_bl:
            # Use a different cropping scheme for the blender dataset because the image size is different
            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=InterpolationMode.BILINEAR)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.25, 1.00), interpolation=InterpolationMode.NEAREST)
            ])
        else:
            self.all_im_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.BILINEAR)
            ])

            self.all_gt_dual_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomResizedCrop((384, 384), scale=(0.36,1.00), interpolation=InterpolationMode.NEAREST)
            ])

        # Final transform without randomness
        self.final_im_transform = transforms.Compose([
            transforms.ToTensor(),
            im_normalization,
        ])

    def __getitem__(self, idx):
        video = self.videos[idx]
        info = {}
        info['name'] = video

        vid_im_path = path.join(self.im_root, video)
        vid_gt_path = path.join(self.gt_root, video)
        frames = self.frames[video]

        trials = 0
        while trials < 5:
            info['frames'] = [] # Appended with actual frames

            num_frames = self.num_frames
            length = len(frames)
            this_max_jump = min(len(frames), self.max_jump)

            # iterative sampling
            frames_idx = [0]
            acceptable_set = set(range(max(1, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1))).difference(set(frames_idx))
            while(len(frames_idx) < num_frames):
                idx = np.random.choice(list(acceptable_set))
                frames_idx.append(idx)
                new_set = set(range(max(1, frames_idx[-1]-this_max_jump), min(length, frames_idx[-1]+this_max_jump+1)))
                acceptable_set = acceptable_set.union(new_set).difference(set(frames_idx))

            frames_idx = sorted(frames_idx)

            sequence_seed = np.random.randint(2147483647)
            images = []
            masks = []
            target_objects = []
            for f_idx in frames_idx:
                jpg_name = frames[f_idx][:-4] + '.jpg'
                png_name = frames[f_idx][:-4] + '.png'
                info['frames'].append(jpg_name)

                reseed(sequence_seed)
                this_im = Image.open(path.join(vid_im_path, jpg_name)).convert('RGB')
                this_im = self.all_im_dual_transform(this_im)
                this_im = self.all_im_lone_transform(this_im)
                reseed(sequence_seed)
                this_gt = Image.open(path.join(vid_gt_path, png_name)).convert('P')
                this_gt = self.all_gt_dual_transform(this_gt)

                pairwise_seed = np.random.randint(2147483647)
                reseed(pairwise_seed)
                this_im = self.pair_im_dual_transform(this_im)
                this_im = self.pair_im_lone_transform(this_im)
                reseed(pairwise_seed)
                this_gt = self.pair_gt_dual_transform(this_gt)

                this_im = self.final_im_transform(this_im)
                this_gt = np.array(this_gt)

                images.append(this_im)
                masks.append(this_gt)

            images = torch.stack(images, 0)

            labels = np.unique(masks[0])
            # Remove background
            labels = labels[labels!=0]

            if self.is_bl:
                # Find large enough labels
                good_lables = []
                for l in labels:
                    pixel_sum = (masks[0]==l).sum()
                    if pixel_sum > 10*10:
                        # OK if the object is always this small
                        # Not OK if it is actually much bigger
                        if pixel_sum > 30*30:
                            good_lables.append(l)
                        elif max((masks[1]==l).sum(), (masks[2]==l).sum()) < 20*20:
                            good_lables.append(l)
                labels = np.array(good_lables, dtype=np.uint8)
            
            if len(labels) == 0:
                target_objects = []
                trials += 1
            else:
                target_objects = labels.tolist()
                break

        if len(target_objects) > self.max_num_obj:
            target_objects = np.random.choice(target_objects, size=self.max_num_obj, replace=False)

        info['num_objects'] = max(1, len(target_objects))

        masks = np.stack(masks, 0)

        # Generate one-hot ground-truth
        cls_gt = np.zeros((self.num_frames, 384, 384), dtype=np.int64)
        first_frame_gt = np.zeros((1, self.max_num_obj, 384, 384), dtype=np.int64)
        for i, l in enumerate(target_objects):
            this_mask = (masks==l)
            cls_gt[this_mask] = i+1
            first_frame_gt[0,i] = (this_mask[0])
        cls_gt = np.expand_dims(cls_gt, 1)

        # 1 if object exist, 0 otherwise
        selector = [1 if i < info['num_objects'] else 0 for i in range(self.max_num_obj)]
        selector = torch.FloatTensor(selector)

        data = {
            'rgb': images,
            'first_frame_gt': first_frame_gt,
            'cls_gt': cls_gt,
            'selector': selector,
            'info': info,
        }

        return data

    def __len__(self):
        return len(self.videos)
    

class DataSeg(DLBase):

    def __init__(self, cfg, split, ignore_labels=[], \
                 root=os.path.expanduser('./dataset'), renorm=False):

        super(DataSeg, self).__init__()

        self.cfg = cfg
        self.root = root
        self.split = split
        self.ignore_labels = ignore_labels

        self._init_palette(cfg["num_classes"])

        # train/val/test splits are pre-cut
        split_fn = os.path.join(self.root, self.split + ".txt")
        assert os.path.isfile(split_fn)

        self.sequence_ids = []
        self.sequence_names = []
        def add_sequence(name):
            vlen = len(self.images)
            assert vlen >= cfg['video_len'], \
                "Detected video shorter [{}] than training length [{}]".format(vlen, \
                                                                cfg["video_len"])
            self.sequence_ids.append(vlen)
            self.sequence_names.append(name)
            return vlen

        self.images = []
        self.masks = []

        token = None
        with open(split_fn, "r") as lines:
            for line in lines:
                _image = line.strip("\n").split(' ')

                _mask = None
                if len(_image) == 2:
                    _image, _mask = _image
                else:
                    assert len(_image) == 1
                    _image = _image[0]

                _image = os.path.join(cfg['dataset_root'], _image.lstrip('/'))
                assert os.path.isfile(_image), '%s not found' % _image
                self.images.append(_image)

                # each sequence may have a different length
                # do some book-keeping e.g. to ensure we have
                # sequences long enough for subsequent sampling
                _token = _image.split("/")[-2] # parent directory
                
                # sequence ID is in the filename
                #_token = os.path.basename(_image).split("_")[0] 
                if token != _token:
                    if not token is None:
                        add_sequence(token)
                    token = _token

                if _mask is None:
                    self.masks.append(None)
                else:
                    _mask = os.path.join(cfg['dataset_root'], _mask.lstrip('/'))
                    assert os.path.isfile(_mask), '%s not found' % _mask
                    self.masks.append(_mask)

        # update the last sequence
        # returns the total amount of frames
        add_sequence(token)
        print("Loaded {} sequences".format(len(self.sequence_ids)))

        # definint data augmentation:
        print("Dataloader: {}".format(split), " #", len(self.images))
        print("\t {}: no augmentation".format(split))

        self.tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=self.MEAN, std=self.STD)])
        self._num_samples = len(self.images)

    def __len__(self):
        return len(self.sequence_ids)

    def denorm(self, image):

        if image.dim() == 3:
            assert image.dim() == 3, "Expected image [CxHxW]"
            assert image.size(0) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip(image, self.MEAN, self.STD):
                t.mul_(s).add_(m)
        elif image.dim() == 4:
            # batch mode
            assert image.size(1) == 3, "Expected RGB image [3xHxW]"

            for t, m, s in zip((0,1,2), self.MEAN, self.STD):
                image[:, t, :, :].mul_(s).add_(m)

        return image
    
    def _mask2tensor(self, mask, num_classes=6):
        h,w = mask.shape
        ones = torch.ones(1,h,w)
        zeros = torch.zeros(num_classes,h,w)
        
        max_idx = mask.max()
        assert max_idx < num_classes, "{} >= {}".format(max_idx, num_classes)
        return zeros.scatter(0, mask[None, ...], ones)

    def __getitem__(self, index):
        seq_to = self.sequence_ids[index] - 1
        seq_from = 0 if index == 0 else self.sequence_ids[index-1] - 1

        info = {}
        info['name'] = self.sequence_names[index]
        info['frames'] = [f_str[-9:] for f_str in self.images[seq_from:seq_to]]

        images, masks, masks_onehot = [], [], []
        n_obj = 0
        for _id_ in range(seq_from, seq_to):

            image = Image.open(self.images[_id_]).convert('RGB')

            if self.masks[_id_] is None:
                mask = Image.new('L', image.size)
            else:
                mask = Image.open(self.masks[_id_]) #.convert('L')

            image = self.tf(image)
            images.append(image)

            mask = torch.from_numpy(np.array(mask, np.int64, copy=False))
            n_obj = max(n_obj, mask.max().item())
            masks.append(mask)
            masks_onehot.append(self._mask2tensor(mask))

        labels = torch.unique(masks[0])
        labels = labels[labels!=0]
        target_objects = labels.tolist()
        max_num_obj = len(target_objects)

        images = torch.stack(images, 0)
        masks = torch.stack(masks, 0)
        masks_onehot = torch.stack(masks_onehot, 0)

        cls_gt = torch.zeros_like(masks)
        first_frame_gt = torch.zeros((1, max_num_obj, *masks.shape[-2:]), dtype=torch.int64)
        for i, l in enumerate(target_objects):
            this_mask = (masks==l)
            cls_gt[this_mask] = i+1
            first_frame_gt[0,i] = (this_mask[0])
        cls_gt = cls_gt.unsqueeze(1)

        info['num_objects'] = n_obj

        # 1 if object exist, 0 otherwise
        selector = [1 if i < info['num_objects'] else 0 for i in range(max_num_obj)]
        selector = torch.FloatTensor(selector)

        data = {
            'rgb': images,
            'first_frame_gt': first_frame_gt,
            'cls_gt': cls_gt,
            'selector': selector,
            'info': info,
            'masks': masks_onehot,
        }

        return data