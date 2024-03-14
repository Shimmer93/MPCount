import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os

from trainers.trainer import Trainer
from utils.misc import denormalize, divide_img_into_patches

class DGTrainer(Trainer):
    def __init__(self, seed, version, device, log_para, patch_size, mode):
        super().__init__(seed, version, device)

        self.log_para = log_para
        self.patch_size = patch_size
        self.mode = mode

    def load_ckpt(self, model, path):
        if isinstance(model, list):
            if path is not None:
                super().load_ckpt(model[0], path[0])
                super().load_ckpt(model[1], path[1])
        else:
            super().load_ckpt(model, path)

    def save_ckpt(self, model, path):
        if isinstance(model, list):
            super().save_ckpt(model[0], path.replace('.pth', '_gen.pth'))
            super().save_ckpt(model[1], path.replace('.pth', '_reg.pth'))
        else:
            super().save_ckpt(model, path)

    def compute_count_loss(self, loss: nn.Module, pred_dmaps, gt_datas, weights=None):
        if loss.__class__.__name__ == 'MSELoss':
            _, gt_dmaps, _ = gt_datas
            gt_dmaps = gt_dmaps.to(self.device)
            if weights is not None:
                pred_dmaps = pred_dmaps * weights
                gt_dmaps = gt_dmaps * weights
            loss_value = loss(pred_dmaps, gt_dmaps * self.log_para)

        elif loss.__class__.__name__ == 'BL':
            gts, targs, st_sizes = gt_datas
            gts = [gt.to(self.device) for gt in gts]
            targs = [targ.to(self.device) for targ in targs]
            st_sizes = st_sizes.to(self.device)
            loss_value = loss(gts, st_sizes, targs, pred_dmaps)

        else:
            raise ValueError('Unknown loss: {}'.format(loss))
        
        return loss_value

    def predict(self, model, img):
        h, w = img.shape[2:]
        ps = self.patch_size
        if h >= ps or w >= ps:
            pred_count = 0
            img_patches, _, _ = divide_img_into_patches(img, ps)
            for patch in img_patches:
                pred = model(patch) if self.mode == 'base' else model(patch)[0]
                pred_count += torch.sum(pred).cpu().item() / self.log_para
        else:
            pred_dmap = model(img) if self.mode == 'base' else model(img)[0]
            pred_count = pred_dmap.sum().cpu().item() / self.log_para

        return pred_count
    
    def predict_isw(self, model, img, img2):
        h, w = img.shape[2:]
        ps = self.patch_size
        if h >= ps or w >= ps:
            pred_count = 0
            img_patches, _, _ = divide_img_into_patches(img, ps)
            img_patches2, _, _ = divide_img_into_patches(img2, ps)
            for patch, patch2 in zip(img_patches, img_patches2):
                pred = model(patch) if self.mode == 'base' else model(patch)[0]
                pred_count += torch.sum(pred).cpu().item() / self.log_para
                model([patch, patch2], cal_covstat=True)
        else:
            pred_dmap = model(img) if self.mode == 'base' else model(img)[0]
            pred_count = pred_dmap.sum().cpu().item() / self.log_para
            model([img, img2], cal_covstat=True)

        return pred_count
    
    def get_visualized_results(self, model, img):
        h, w = img.shape[2:]
        ps = self.patch_size
        if h >= ps or w >= ps:
            dmap = torch.zeros(1, 1, h, w)
            img_patches, nh, nw = divide_img_into_patches(img, ps)
            for i in range(nh):
                for j in range(nw):
                    patch = img_patches[i*nw+j]
                    pred_dmap = model(patch)
                    dmap[:, :, i*ps:(i+1)*ps, j*ps:(j+1)*ps] = pred_dmap
        else:
            dmap = model(img)

        dmap = dmap[0, 0].cpu().detach().numpy().squeeze()

        return dmap
    
    def get_visualized_results_with_cls(self, model, img):
        h, w = img.shape[2:]
        ps = self.patch_size
        if h >= ps or w >= ps:
            dmap = torch.zeros(1, 1, h, w)
            cmap = torch.zeros(1, 3, h//16, w//16)
            img_patches, nh, nw = divide_img_into_patches(img, ps)
            for i in range(nh):
                for j in range(nw):
                    patch = img_patches[i*nw+j]
                    pred_dmap, pred_cmap = model(patch)
                    dmap[:, :, i*ps:(i+1)*ps, j*ps:(j+1)*ps] = pred_dmap
                    cmap[:, :, i*ps//16:(i+1)*ps//16, j*ps//16:(j+1)*ps//16] = pred_cmap
        else:
            dmap, cmap = model(img)

        dmap = dmap[0, 0].cpu().detach().numpy().squeeze()
        cmap = cmap[0, 0].cpu().detach().numpy().squeeze()

        return dmap, cmap
    
    def train_step(self, model, loss, optimizer, batch, epoch):
        imgs1, imgs2, gt_datas = batch
        imgs1 = imgs1.to(self.device)
        imgs2 = imgs2.to(self.device)
        gt_cmaps = gt_datas[-1].to(self.device)

        if self.mode == 'simple':
            optimizer.zero_grad()
            dmaps1 = model(imgs1)
            loss_den = self.compute_count_loss(loss, dmaps1, gt_datas)
            loss_total = loss_den
            loss_total.backward()
            optimizer.step()

        elif self.mode == 'base':
            optimizer.zero_grad()
            dmaps1 = model(imgs1)
            dmaps2 = model(imgs2)
            loss_den = self.compute_count_loss(loss, dmaps1, gt_datas) + self.compute_count_loss(loss, dmaps2, gt_datas)
            loss_total = loss_den
            loss_total.backward()
            optimizer.step()

        elif self.mode == 'add':
            optimizer.zero_grad()
            dmaps1, dmaps2, loss_con = model.forward_train(imgs1, imgs2)
            loss_den = self.compute_count_loss(loss, dmaps1, gt_datas) + self.compute_count_loss(loss, dmaps2, gt_datas)
            loss_total = loss_den + loss_con
            loss_total.backward()
            optimizer.step()

        elif self.mode == 'cls':
            optimizer.zero_grad()
            dmaps1, cmaps1 = model(imgs1, gt_cmaps)
            dmaps2, cmaps2 = model(imgs2, gt_cmaps)
            loss_den = self.compute_count_loss(loss, dmaps1, gt_datas) + self.compute_count_loss(loss, dmaps2, gt_datas)
            loss_cls = F.binary_cross_entropy(cmaps1, gt_cmaps) + F.binary_cross_entropy(cmaps2, gt_cmaps)
            loss_total = loss_den + 10 * loss_cls
            loss_total.backward()
            optimizer.step()

        elif self.mode == 'final':
            optimizer.zero_grad()
            dmaps1, dmaps2, cmaps1, cmaps2, cerrmap, loss_con, loss_err = model.forward_train(imgs1, imgs2, gt_cmaps)
            loss_den = self.compute_count_loss(loss, dmaps1, gt_datas) + self.compute_count_loss(loss, dmaps2, gt_datas)
            loss_cls = F.binary_cross_entropy(cmaps1, gt_cmaps) + F.binary_cross_entropy(cmaps2, gt_cmaps)
            loss_total = loss_den + 10 * loss_cls + 10 * loss_con # + loss_err 

            loss_total.backward()
            optimizer.step()

        elif self.mode == 'isw':
            optimizer.zero_grad()
            gts = gt_datas[1].to(self.device)
            losses = model(imgs1, gts=gts, apply_wtloss=(epoch>5))
            loss_total = torch.FloatTensor([0]).cuda()
            loss_total += losses[0]
            # loss_total += 0.4 * losses[1]
            if epoch > 5:
                loss_total += 0.6 * losses[1]
            loss_total.backward()
            optimizer.step()
            
        else:
            raise ValueError('Unknown mode: {}'.format(self.mode))

        return loss_total.detach().item()

    def val_step(self, model, batch):
            
        img1, img2, gt, _, _ = batch
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        if self.mode == 'isw':
            with torch.no_grad():
                pred_count = self.predict_isw(model, img1, img2)
        else:
            pred_count = self.predict(model, img1)
        gt_count = gt.shape[1]
        mae = np.abs(pred_count - gt_count)
        mse = (pred_count - gt_count) ** 2

        return mae, {'mse': mse}

    def test_step(self, model, batch):
        img1, _, gt, _, _ = batch
        img1 = img1.to(self.device)
        # img2 = img2.to(self.device)

        pred_count = self.predict(model, img1)
        gt_count = gt.shape[1]
        mae = np.abs(pred_count - gt_count)
        mse = (pred_count - gt_count) ** 2
        return {'mae': mae, 'mse': mse}
        
    def vis_step(self, model, batch):
        img1, img2, gt, name, _ = batch
        vis_dir = os.path.join(self.log_dir, 'vis')
        img1 = img1.to(self.device)
        img2 = img2.to(self.device)

        if self.mode == 'base':
            pred_dmap1 = self.get_visualized_results(model, img1)
            pred_dmap2 = self.get_visualized_results(model, img2)
            img1 = denormalize(img1.detach())[0].cpu().permute(1, 2, 0).numpy()
            img2 = denormalize(img2.detach())[0].cpu().permute(1, 2, 0).numpy()
            pred_count1 = pred_dmap1.sum() / self.log_para
            pred_count2 = pred_dmap2.sum() / self.log_para
            gt_count = gt.shape[1]

            datas = [img1, pred_dmap1, img2, pred_dmap2]
            titles = [name, f'Pred1: {pred_count1}', f'GT: {gt_count}', f'Pred2: {pred_count2}']

            fig = plt.figure(figsize=(10, 6))
            for i in range(4):
                ax = fig.add_subplot(2, 2, i+1)
                ax.set_title(titles[i])
                ax.imshow(datas[i])

            plt.savefig(os.path.join(vis_dir, f'{name[0]}.png'))
            plt.close()

        else:
            pred_dmap1, pred_cmap1 = self.get_visualized_results_with_cls(model, img1)
            pred_dmap2, pred_cmap2 = self.get_visualized_results_with_cls(model, img2)
            img1 = denormalize(img1.detach())[0].cpu().permute(1, 2, 0).numpy()
            img2 = denormalize(img2.detach())[0].cpu().permute(1, 2, 0).numpy()
            pred_count1 = pred_dmap1.sum() / self.log_para
            pred_count2 = pred_dmap2.sum() / self.log_para
            gt_count = gt.shape[1]

            new_cmap1 = pred_cmap1.copy()
            new_cmap1[new_cmap1 < 0.5] = 0
            new_cmap1[new_cmap1 >= 0.5] = 1
            # new_cmap1 = np.resize(new_cmap1, (new_cmap1.shape[0]*16, new_cmap1.shape[1]*16))
            new_cmap2 = pred_cmap2.copy()
            new_cmap2[new_cmap2 < 0.5] = 0
            new_cmap2[new_cmap2 >= 0.5] = 1
            # new_cmap2 = np.resize(new_cmap2, (new_cmap2.shape[0]*16, new_cmap2.shape[1]*16))

            datas = [img1, pred_dmap1, pred_cmap1, img2, pred_dmap2, pred_cmap2]
            titles = [name[0], f'Pred1: {pred_count1}', 'Cls1', f'GT: {gt_count}', f'Pred2: {pred_count2}', 'Cls2']

            fig = plt.figure(figsize=(15, 6))
            for i in range(6):
                ax = fig.add_subplot(2, 3, i+1)
                ax.set_title(titles[i])
                ax.imshow(datas[i])
            plt.savefig(os.path.join(vis_dir, f'{name[0]}.png'))

            new_datas = [img1, pred_cmap1, new_cmap1, pred_dmap1]
            new_titles = [f'{name[0]}', 'Cls', 'BCls', f'Pred_{pred_count1}']
            for i in range(len(new_datas)):
                plt.imsave(os.path.join(vis_dir, f'{name[0]}_{new_titles[i]}.png'), new_datas[i])

            plt.close()