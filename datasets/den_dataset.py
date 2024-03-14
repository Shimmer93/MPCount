import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image

import random
import os

import sys
sys.path.append('..')

from datasets.base_dataset import BaseDataset
from utils.misc import random_crop, get_padding

class DensityMapDataset(BaseDataset):
    def collate(batch):
        transposed_batch = list(zip(*batch))
        images = torch.stack(transposed_batch[0], 0)
        points = transposed_batch[1]  # the number of points is not fixed, keep it as a list of tensor
        dmaps = torch.stack(transposed_batch[2], 0)
        return images, (points, dmaps)

    def __init__(self, root, crop_size, downsample, method, is_grey, unit_size, pre_resize=1, roi_map_path=None, gt_dir=None, gen_root=None):
        super().__init__(root, crop_size, downsample, method, is_grey, unit_size, pre_resize, roi_map_path, gen_root)
        self.gt_dir = gt_dir

    def _load_dmap(self, dmap_fn):
        dmap = np.load(dmap_fn)
        if self.roi_map is not None:
            dmap = dmap * self.roi_map.astype(np.float32)
        return dmap
    
    def __getitem__(self, index):
        img_fn = self.img_fns[index]
        img, img_ext = self._load_img(img_fn)

        basename = img_fn.split('/')[-1].split('.')[0]
        if img_fn.startswith(self.root):
            gt_fn = img_fn.replace(img_ext, '.npy')
        else:
            basename = basename[:-2]
            gt_fn = os.path.join(self.root, 'train', basename + '.npy')
        gt = self._load_gt(gt_fn)

        if self.method == 'train':
            if self.gt_dir is None:
                dmap_fn = gt_fn.replace(basename, basename + '_dmap2')
            else:
                dmap_fn = os.path.join(self.gt_dir, basename + '.npy')
            dmap = self._load_dmap(dmap_fn)
            return tuple(self._train_transform(img, gt, dmap))
        elif self.method in ['val', 'test']:
            return tuple(self._val_transform(img, gt, basename))

    def _train_transform(self, img, gt, dmap):
        w, h = img.size
        assert len(gt) >= 0

        dmap = torch.from_numpy(dmap).unsqueeze(0)

        # Grey Scale
        if random.random() > 0.88:
            img = img.convert('L').convert('RGB')

        # Resizing
        # factor = random.random() * 0.5 + 0.75
        factor = self.pre_resize * (random.random() * 0.5 + 0.75)
        if factor != 1.0:
            new_w = (int)(w * factor)
            new_h = (int)(h * factor)
            # if min(new_w, new_h) >= min(self.crop_size[0], self.crop_size[1]):
            w = new_w
            h = new_h
            img = img.resize((w, h))
            dmap_sum = dmap.sum()
            dmap = F.resize(dmap, (h, w))
            new_dmap_sum = dmap.sum()
            dmap = dmap * dmap_sum / new_dmap_sum
            gt = gt * factor
        
        # Padding
        st_size = 1.0 * min(w, h)
        if st_size < min(self.crop_size[0], self.crop_size[1]):
            padding, h, w = get_padding(h, w, self.crop_size[0], self.crop_size[1])
            left, top, _, _ = padding

            img = F.pad(img, padding)
            dmap = F.pad(dmap, padding)
            if len(gt) > 0:
                gt = gt + [left, top]

        # Cropping
        i, j = random_crop(h, w, self.crop_size[0], self.crop_size[1])
        h, w = self.crop_size[0], self.crop_size[1]
        img = F.crop(img, i, j, h, w)
        h, w = self.crop_size[0], self.crop_size[1]
        dmap = F.crop(dmap, i, j, h, w)
        h, w = self.crop_size[0], self.crop_size[1]

        if len(gt) > 0:
            gt = gt - [j, i]
            idx_mask = (gt[:, 0] >= 0) * (gt[:, 0] <= w) * \
                       (gt[:, 1] >= 0) * (gt[:, 1] <= h)
            gt = gt[idx_mask]
        else:
            gt = np.empty([0, 2])

        # Downsampling
        down_w = w // self.downsample
        down_h = h // self.downsample
        dmap = dmap.reshape([1, down_h, self.downsample, down_w, self.downsample]).sum(dim=(2, 4))

        if len(gt) > 0:
            gt = gt / self.downsample

        # Flipping
        if random.random() > 0.5:
            img = F.hflip(img)
            dmap = F.hflip(dmap)
            if len(gt) > 0:
                gt[:, 0] = w - gt[:, 0]
        
        # Post-processing
        img = self.transform(img)
        gt = torch.from_numpy(gt.copy()).float()
        dmap = dmap.float()

        return img, gt, dmap

if __name__ == '__main__':
    dataset = DensityMapDataset('/mnt/home/zpengac/USERDIR/Crowd_counting/datasets/sta', 16, 1, 'train', False, 1)

    count = 0
    has_crowd = 0
    for i in range(len(dataset.img_fns)):
        img, gt, dmap = dataset[i]
        if len(gt) > 0:
            has_crowd += 1
        count += 1

    print(count, has_crowd)