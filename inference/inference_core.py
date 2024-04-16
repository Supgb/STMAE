from inference.inference_memory_bank import MemoryBank
from model.network import STMAE
from model.aggregate import aggregate

from util.tensor_util import pad_divide_by, unpad
import torch.nn.functional as F


class InferenceCore:
    def __init__(self, network:STMAE, config):
        self.config = config
        self.network = network
        self.mem_every = config['mem_every']

        self.clear_memory()
        self.all_labels = None

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = 0
        self.memory = MemoryBank(top_k=self.config['top_k'])

    def update_config(self, config):
        self.mem_every = config['mem_every']

    def set_all_labels(self, all_labels):
        # self.all_labels = [l.item() for l in all_labels]
        self.all_labels = all_labels

    def step(self, image, mask=None, valid_labels=None, end=False):
        # image: 3*H*W
        # mask: num_objects*H*W or None
        self.curr_ti += 1
        image, self.pad = pad_divide_by(image, 16)
        image = image.unsqueeze(0) # add the batch dimension

        is_mem_frame = ((self.curr_ti-self.last_mem_ti >= self.mem_every) or (mask is not None)) and (not end)
        need_segment = (self.curr_ti > 0) and ((valid_labels is None) or (len(self.all_labels) != len(valid_labels)))

        f16, f8, f4 = self.network.encode_key(image)
        key = f16
        multi_scale_features = (f16, f8, f4)

        # segment the current frame is needed
        if need_segment:
            memory_readout = self.memory.match_memory(key)
            memory_readout = F.interpolate(memory_readout, size=image.shape[-2:], mode='bilinear', align_corners=False)
            obj_emb, _ = self.network.encode_value(image, [f16], None, memory_readout)
            _, _, _, pred_prob_with_bg = self.network.segment(multi_scale_features, obj_emb, 
                                    hidden_state=None, h_out=None, strip_bg=True)
            # remove batch dim
            pred_prob_with_bg = pred_prob_with_bg[0]
            pred_prob_no_bg = pred_prob_with_bg[1:]
        else:
            pred_prob_no_bg = pred_prob_with_bg = None

        # use the input mask if any
        if mask is not None:
            mask, _ = pad_divide_by(mask, 16)

            if pred_prob_no_bg is not None:
                # if we have a predicted mask, we work on it
                # make pred_prob_no_bg consistent with the input mask
                mask_regions = (mask.sum(0) > 0.5)
                pred_prob_no_bg[:, mask_regions] = 0
                # shift by 1 because mask/pred_prob_no_bg do not contain background
                mask = mask.type_as(pred_prob_no_bg)
                if valid_labels is not None:
                    shift_by_one_non_labels = [i for i in range(pred_prob_no_bg.shape[0]) if (i+1) not in valid_labels]
                    # non-labelled objects are copied from the predicted mask
                    mask[shift_by_one_non_labels] = pred_prob_no_bg[shift_by_one_non_labels]
            pred_prob_with_bg = aggregate(mask, dim=0)

        # save as memory if needed
        if is_mem_frame:
            value = F.interpolate(pred_prob_with_bg[1:].unsqueeze(1), 
                                  size=key.shape[-2:], mode='bilinear', align_corners=False)
            self.memory.add_memory(key, value, self.all_labels)
            self.last_mem_ti = self.curr_ti
                
        return unpad(pred_prob_with_bg, self.pad)
