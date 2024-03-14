import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import torchvision.transforms.functional as F
from glob import glob
import numpy as np
from PIL import Image

import random
import os

import sys
sys.path.append('..')

from utils.misc import random_crop, get_padding

class BaseDataset(Dataset):

    def __init__(self, root, crop_size, downsample, method, is_grey=False, unit_size=0, pre_resize=1, roi_map_path=None, gen_root=None):
        self.root = root
        self.gen_root = gen_root
        if isinstance(crop_size, int):
            self.crop_size = (crop_size, crop_size)
        else:
            self.crop_size = crop_size
        self.downsample = downsample
        self.method = method
        self.is_grey = is_grey
        self.unit_size = unit_size
        self.pre_resize = pre_resize
        self.roi_map = np.load(roi_map_path, allow_pickle=True).tolist() if roi_map_path is not None else None

        if self.method == 'train':
            self.transform = T.Compose([
                # T.RandomApply([T.ColorJitter(brightness=0.5, contrast=0.3, saturation=0.3, hue=0.1)], p=0.8),
                # T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=1)], p=0.2),
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        else:
            self.transform = T.Compose([
                T.ToTensor(),
                T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
                # T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

        if self.method not in ['train', 'val', 'test']:
            raise ValueError('method must be train, val or test')
        self.img_fns = glob(os.path.join(root, self.method, '*.jpg')) + \
                          glob(os.path.join(root, self.method, '*.png'))
        if self.gen_root is not None and self.method == 'train':
            self.img_fns += glob(os.path.join(self.gen_root, '*.jpg')) + \
                              glob(os.path.join(self.gen_root, '*.png'))
            
        if self.method in ['val', 'test']:
            self.img_fns = sorted(self.img_fns)
        
    def __len__(self):
        return len(self.img_fns)

    def _load_img(self, img_fn):
        img = Image.open(img_fn).convert('RGB')
        if self.roi_map is not None:
            img_array = np.array(img)
            img_array = img_array * np.expand_dims(self.roi_map, axis=2)
            img = Image.fromarray(img_array)
        img_ext = os.path.splitext(img_fn)[1]
        return img, img_ext

    def _load_gt(self, gt_fn):
        gt = np.load(gt_fn)
        if self.roi_map is not None:
            gt = gt[np.where(self.roi_map[gt[:, 1].astype(int), gt[:, 0].astype(int)])]
        return gt

    def __getitem__(self, index):
        img_fn = self.img_fns[index]
        name = img_fn.split('/')[-1].split('.')[0]
        img, img_ext = self._load_img(img_fn)
        if img_fn.startswith(self.root):
            gt_fn = img_fn.replace(img_ext, '.npy')
        else:
            gt_fn = os.path.join(self.root, 'train', name[:-2] + '.npy')
        gt = self._load_gt(gt_fn)
        
        if self.method == 'train':
            return tuple(self._train_transform(img, gt))
        elif self.method in ['val', 'test']:
            return tuple(self._val_transform(img, gt, name))

    def _train_transform(self, img, gt):
        w, h = img.size
        assert len(gt) >= 0

        # Grey Scale
        if random.random() > 0.88:
            img = img.convert('L').convert('RGB')

        # Resizing

        factor = (random.random() * 0.5 + 0.75) * self.pre_resize
        new_w = (int)(w * factor)
        new_h = (int)(h * factor)
        if min(new_w, new_h) >= min(self.crop_size[0], self.crop_size[1]):
            w = new_w
            h = new_h
            img = img.resize((w, h))
            gt = gt * factor
        
        # Padding
        st_size = 1.0 * min(w, h)
        if st_size < min(self.crop_size[0], self.crop_size[1]):
            padding, h, w = get_padding(h, w, self.crop_size[0], self.crop_size[1])
            left, top, _, _ = padding

            img = F.pad(img, padding)
            gt = gt + [left, top]

        # Cropping
        i, j = random_crop(h, w, self.crop_size[0], self.crop_size[1])
        h, w = self.crop_size[0], self.crop_size[1]
        img = F.crop(img, i, j, h, w)
        h, w = self.crop_size[0], self.crop_size[1]

        if len(gt) > 0:
            gt = gt - [j, i]
            idx_mask = (gt[:, 0] >= 0) * (gt[:, 0] <= w) * \
                       (gt[:, 1] >= 0) * (gt[:, 1] <= h)
            gt = gt[idx_mask]
        else:
            gt = np.empty([0, 2])

        # Downsampling
        gt = gt / self.downsample

        # Flipping
        if random.random() > 0.5:
            img = F.hflip(img)
            if len(gt) > 0:
                gt[:, 0] = w - gt[:, 0]
        
        # Post-processing
        img = self.transform(img)
        gt = torch.from_numpy(gt.copy()).float()

        return img, gt

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
        img = self.transform(img)
        gt = torch.from_numpy(gt.copy()).float()

        return img, gt, name, padding
    
if __name__ == '__main__':
    from torch.utils.data import DataLoader

    dataset = BaseDataset(root='/mnt/home/zpengac/USERDIR/Crowd_counting/datasets/sta', 
                          crop_size=512, downsample=1, method='train', unit_size=16,
                          gen_root='/mnt/home/zpengac/USERDIR/Crowd_counting/DGVCC/logs/sta_joint_nocheat2/gen')
    # dataloader = DataLoader(dataset, batch_size=2, shuffle=False, num_workers=0)
    # for i, (img, gt, name, padding) in enumerate(dataloader):
    #     print(name, padding)