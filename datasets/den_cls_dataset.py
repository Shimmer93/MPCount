import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image

import random
import os

import sys
sys.path.append('..')

from datasets.den_dataset import DensityMapDataset
from utils.misc import random_crop, get_padding

class DenClsDataset(DensityMapDataset):
    def collate(batch):
        transposed_batch = list(zip(*batch))
        images1 = torch.stack(transposed_batch[0], 0)
        images2 = torch.stack(transposed_batch[1], 0)
        points = transposed_batch[2]  # the number of points is not fixed, keep it as a list of tensor
        dmaps = torch.stack(transposed_batch[3], 0)
        bmaps = torch.stack(transposed_batch[4], 0)
        return images1, images2, (points, dmaps, bmaps)

    def __init__(self, root, crop_size, downsample, method, is_grey, unit_size, pre_resize=1, roi_map_path=None, gt_dir=None, gen_root=None):
        super().__init__(root, crop_size, downsample, method, is_grey, unit_size, pre_resize, roi_map_path, gt_dir, gen_root)

        self.more_transform = T.Compose([
            T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.2, saturation=0.2, hue=0.1)], p=0.8),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=1)], p=0.5),
            T.RandomAdjustSharpness(sharpness_factor=5, p=0.5),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def __getitem__(self, index):
        img_fn = self.img_fns[index]
        img, img_ext = self._load_img(img_fn)

        basename = img_fn.split('/')[-1].split('.')[0]
        if img_fn.startswith(self.root):
            gt_fn = img_fn.replace(img_ext, '.npy')
            if basename.endswith('_aug'):
                gt_fn = gt_fn.replace('_aug', '')
            elif basename.endswith('_aug2'):
                gt_fn = gt_fn.replace('_aug2', '')
        else:
            basename = basename[:-2]
            gt_fn = os.path.join(self.root, 'train', basename + '.npy')
        gt = self._load_gt(gt_fn)

        if self.method == 'train':
            if self.gt_dir is None:
                dmap_fn = gt_fn.replace(basename, basename + '_dmap')
            else:
                dmap_fn = os.path.join(self.gt_dir, basename + '.npy')
            dmap = self._load_dmap(dmap_fn)
            img1, img2, gt, dmap = self._train_transform(img, gt, dmap)
            # bmap_orig = dmap.clone().reshape(1, dmap.shape[1]//32, 32, dmap.shape[2]//32, 32).sum(dim=(2, 4))
            # bmap = ((bmap_orig > 0).long() + (bmap_orig > 1).long()).squeeze(0)
            bmap_orig = dmap.clone().reshape(1, dmap.shape[1]//16, 16, dmap.shape[2]//16, 16).sum(dim=(2, 4))
            bmap = (bmap_orig > 0).float()
            return img1, img2, gt, dmap, bmap
        elif self.method in ['val', 'test']:
            return self._val_transform(img, gt, basename)
        
    def _rotate_gt(self, gt, w, h, angle):
        gt = np.array(gt)
        gt = gt - [w/2, h/2]
        theta = angle / 180.0 * np.pi
        rot_mat = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        gt = np.dot(gt, rot_mat)
        gt = gt + [w/2, h/2]
        return gt

    def _train_transform(self, img, gt, dmap):
        w, h = img.size
        assert len(gt) >= 0

        dmap = torch.from_numpy(dmap).unsqueeze(0)

        # Grey Scale
        if random.random() > 0.88:
            img = img.convert('L').convert('RGB')

        # Resizing
        # factor = random.random() * 0.5 + 0.75
        # factor = self.pre_resize * (random.random() * 0.5 + 0.75)
        # if factor != 1.0:
        #     new_w = (int)(w * factor)
        #     new_h = (int)(h * factor)
        #     if min(new_w, new_h) >= min(self.crop_size[0], self.crop_size[1]):
        #         w = new_w
        #         h = new_h
        #         img = img.resize((w, h))
        #         dmap_sum = dmap.sum()
        #         dmap = F.resize(dmap, (h, w))
        #         new_dmap_sum = dmap.sum()
        #         dmap = dmap * dmap_sum / new_dmap_sum
        #         gt = gt * factor

        # Rotation
        # if random.random() > 0.5:
        #     angle = random.random() * 20 - 10
        #     img = F.rotate(img, angle, F.InterpolationMode.BILINEAR)
        #     dmap = F.rotate(dmap, angle, F.InterpolationMode.BILINEAR)
            # gt = self._rotate_gt(gt, w, h, angle)
        
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
        img1 = self.transform(img)
        img2 = self.more_transform(img)
        gt = torch.from_numpy(gt.copy()).float()
        dmap = dmap.float()

        return img1, img2, gt, dmap

    def _val_transform(self, img, gt, name):
        if self.pre_resize != 1:
            img = img.resize((int(img.size[0] * self.pre_resize), int(img.size[1] * self.pre_resize)))

        if self.unit_size is not None and self.unit_size > 0:
            # Padding
            w, h = img.size
            new_w = (w // self.unit_size + 1) * self.unit_size if w % self.unit_size != 0 else w
            new_h = (h // self.unit_size + 1) * self.unit_size if h % self.unit_size != 0 else h

            padding, h, w = get_padding(h, w, new_h, new_w)
            left, top, _, _ = padding

            img = F.pad(img, padding)
            if len(gt) > 0:
                gt = gt + [left, top]
        else:
            padding = (0, 0, 0, 0)

        # Downsampling
        gt = gt / self.downsample

        # Post-processing
        img1 = self.transform(img)
        img2 = self.more_transform(img)
        gt = torch.from_numpy(gt.copy()).float()

        return img1, img2, gt, name, padding
        
if __name__ == '__main__':
    dataset = DenClsDataset('/mnt/home/zpengac/USERDIR/Crowd_counting/datasets/sta', 320, 1, 'train', False, 1)

    count0 = 0
    count1 = 0
    for i in range(len(dataset)):
        img, _, gt, dmap, bmap = dataset[i]
        # print(bmap)
        # break
        count0 += (bmap == 0).sum()
        count1 += (bmap == 1).sum()

    print(count0, count1)