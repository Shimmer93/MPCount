import numpy as np
import torch
from rich.progress import track

import random
import os
import time

def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j

def get_padding(h, w, new_h, new_w):
    if h >= new_h:
        top = 0
        bottom = 0
    else:
        dh = new_h - h
        top = dh // 2
        bottom = dh // 2 + dh % 2
        h = new_h
    if w >= new_w:
        left = 0
        right = 0
    else:
        dw = new_w - w
        left = dw // 2
        right = dw // 2 + dw % 2
        w = new_w
    
    return (left, top, right, bottom), h, w

def cal_inner_area(c_left, c_up, c_right, c_down, bbox):
    inner_left = np.maximum(c_left, bbox[:, 0])
    inner_up = np.maximum(c_up, bbox[:, 1])
    inner_right = np.minimum(c_right, bbox[:, 2])
    inner_down = np.minimum(c_down, bbox[:, 3])
    inner_area = np.maximum(inner_right-inner_left, 0.0) * np.maximum(inner_down-inner_up, 0.0)
    return inner_area

def divide_img_into_patches(img, patch_size):
    h, w = img.shape[-2:]

    img_patches = []
    h_stride = int(np.ceil(1.0 * h / patch_size))
    w_stride = int(np.ceil(1.0 * w / patch_size))
    for i in range(h_stride):
        for j in range(w_stride):
            h_start = i * patch_size
            if i != h_stride - 1:
                h_end = (i + 1) * patch_size
            else:
                h_end = h
            w_start = j * patch_size
            if j != w_stride - 1:
                w_end = (j + 1) * patch_size
            else:
                w_end = w
            img_patches.append(img[..., h_start:h_end, w_start:w_end])

    return img_patches, h_stride, w_stride

def denormalize(img_tensor):
    # denormalize a image tensor
    if len(img_tensor.shape) == 3:
        img_tensor = img_tensor * torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(img_tensor.device)
        img_tensor = img_tensor + torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1).to(img_tensor.device)
    elif len(img_tensor.shape) == 4:
        img_tensor = img_tensor * torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(img_tensor.device)
        img_tensor = img_tensor + torch.tensor([0.5, 0.5, 0.5]).view(1, 3, 1, 1).to(img_tensor.device)
    # img_tensor = img_tensor * torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_tensor.device)
    # img_tensor = img_tensor + torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_tensor.device)
    return img_tensor

def denormalize2(img_tensor):
    # denormalize a image tensor
    img_tensor = (img_tensor - img_tensor.min() / (img_tensor.max() - img_tensor.min()))
    return img_tensor

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class DictAvgMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = {}
        self.avg = {}
        self.sum = {}
        self.count = {}

    def update(self, val, n=1):
        for k, v in val.items():
            if k not in self.val:
                self.val[k] = 0
                self.sum[k] = 0
                self.count[k] = 0
            self.val[k] = v
            self.sum[k] += v * n
            self.count[k] += n
            self.avg[k] = self.sum[k] / self.count[k]

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.deterministic = False

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_seeded_generator(seed):
    g = torch.Generator()
    g.manual_seed(0)
    return g
    
def get_current_datetime():
    return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())

def easy_track(iterable, description=None):
    return track(iterable, description=description, complete_style='dim cyan', total=len(iterable))
