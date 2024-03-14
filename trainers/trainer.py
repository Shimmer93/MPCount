import torch
import torch.nn as nn
import os
import time
from glob import glob

from utils.misc import AverageMeter, DictAvgMeter, get_current_datetime, easy_track

class Trainer(object):
    def __init__(self, seed, version, device):

        self.seed = seed
        self.version = version
        self.device = torch.device(device)

        self.log_dir = os.path.join('logs', self.version)
        os.makedirs(self.log_dir, exist_ok=True)

    def log(self, msg, verbose=True, **kwargs):
        if verbose:
            print(msg, **kwargs)
        with open(os.path.join(self.log_dir, 'log.txt'), 'a') as f:
            if 'end' in kwargs:
                f.write(msg + kwargs['end'])
            else:
                f.write(msg + '\n')

    def load_ckpt(self, model, path):
        if path is not None:
            self.log('Loading checkpoint from {}'.format(path))
            model.load_state_dict(torch.load(path, map_location=self.device), strict=False)

    def save_ckpt(self, model, path):
        torch.save(model.state_dict(), path)

    def set_model_train(self, model):
        if isinstance(model, nn.Module):
            model.train()
        else:
            model[0].train()
            model[1].train()

    def set_model_eval(self, model):
        if isinstance(model, nn.Module):
            model.eval()
        else:
            model[0].eval()
            model[1].eval()

    def train_step(self, model, loss, optimizer, batch, epoch):
        pass

    def val_step(self, model, batch):
        pass

    def test_step(self, model, batch):
        pass

    def vis_step(self, model, batch):
        pass

    def train_epoch(self, model, loss, train_dataloader, val_dataloader, optimizer, scheduler, epoch, best_criterion, best_epoch):
        start_time = time.time()

        # Training
        self.set_model_train(model)
        for batch in easy_track(train_dataloader, description=f'Epoch {epoch}: Training...'):
            train_loss = self.train_step(model, loss, optimizer, batch, epoch)
        if scheduler is not None:
            if isinstance(scheduler, list):
                for s in scheduler:
                    s.step()
            else:
                scheduler.step()
        self.log(f'Epoch {epoch}: Training loss: {train_loss:.4f} Version: {self.version}')

        # Validation
        self.set_model_eval(model)
        criterion_meter = AverageMeter()
        additional_meter = DictAvgMeter()
        for batch in easy_track(val_dataloader, description=f'Epoch {epoch}: Validating...'):
            with torch.no_grad():
                criterion, additional = self.val_step(model, batch)
            criterion_meter.update(criterion, additional['n']) if 'n' in additional else criterion_meter.update(criterion)
            additional_meter.update(additional)
        current_criterion = criterion_meter.avg
        self.log(f'Epoch {epoch}: Val criterion: {current_criterion:.4f}', end=' ')
        for k, v in additional_meter.avg.items():
            self.log(f'{k}: {v:.4f}', end=' ')
        self.log(f'best: {best_criterion:.4f}, time: {time.time() - start_time:.4f}')

        # Checkpoint
        if epoch > 0:
            os.remove(glob(os.path.join(self.log_dir, 'last*.pth'))[0])
        self.save_ckpt(model, os.path.join(self.log_dir, f'last.pth'))
        if current_criterion < best_criterion:
            best_criterion = current_criterion
            best_epoch = epoch
            self.log(f'Epoch {epoch}: saving best model...')
            if epoch > 0:
                os.remove(glob(os.path.join(self.log_dir, 'best*.pth'))[0])
            self.save_ckpt(model, os.path.join(self.log_dir, f'best.pth'))

        return best_criterion, best_epoch
        
    def train(self, model, loss, train_dataloader, val_dataloader, optimizer, scheduler, checkpoint=None, num_epochs=100):
        self.log('Start training at {}'.format(get_current_datetime()))
        self.load_ckpt(model, checkpoint)

        model = model.to(self.device) if isinstance(model, nn.Module) else [m.to(self.device) for m in model]
        loss = loss.to(self.device)
        
        best_criterion = 1e10
        best_epoch = -1

        for epoch in range(num_epochs):
            best_criterion, best_epoch = self.train_epoch(
                model, loss, train_dataloader, val_dataloader, optimizer, scheduler, epoch, best_criterion, best_epoch)

        self.log('Best epoch: {}, best criterion: {}'.format(best_epoch, best_criterion))
        self.log('Training results saved to {}'.format(self.log_dir))
        self.log('End training at {}'.format(get_current_datetime()))

    def test(self, model, test_dataloader, checkpoint=None):
        self.log('Start testing at {}'.format(get_current_datetime()))
        self.load_ckpt(model, checkpoint)

        model = model.to(self.device) if isinstance(model, nn.Module) else [m.to(self.device) for m in model]

        self.set_model_eval(model)
        result_meter = DictAvgMeter()
        for batch in easy_track(test_dataloader, description='Testing...'):
            with torch.no_grad():
                result = self.test_step(model, batch)
            result_meter.update(result)
        self.log('Testing results:', end=' ')
        for key, value in result_meter.avg.items():
            self.log('{}: {:.4f}'.format(key, value), end=' ')
        self.log('')

        self.log('Testing results saved to {}'.format(self.log_dir))
        self.log('End testing at {}'.format(get_current_datetime()))

    def vis(self, model, test_dataloader, checkpoint=None):
        self.log('Start visualization at {}'.format(get_current_datetime()))
        self.load_ckpt(model, checkpoint)

        os.makedirs(os.path.join(self.log_dir, 'vis'), exist_ok=True)

        model = model.to(self.device) if isinstance(model, nn.Module) else [m.to(self.device) for m in model]

        self.set_model_eval(model)
        for batch in easy_track(test_dataloader, description='Visualizing...'):
            with torch.no_grad():
                self.vis_step(model, batch)

        self.log('Visualization results saved to {}'.format(self.log_dir))
        self.log('End visualization at {}'.format(get_current_datetime()))