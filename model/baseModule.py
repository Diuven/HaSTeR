import torch
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule
from pytorch_lightning.metrics.classification import Accuracy
from abc import ABC, abstractmethod
from dataset import KSXDataset


class BaseModule(LightningModule, ABC):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp
        self.save_hyperparameters()

    @abstractmethod
    def get_loss(self, batch, infer):
        pass

    @abstractmethod
    def get_pred(self, infer):
        pass

    def get_acc(self, batch, infer):
        img, targ = batch
        pred = self.get_pred(infer)
        acc = pred.eq(targ).to(dtype=torch.float).mean()
        return acc

    def training_step(self, batch, index, optimizer_idx=0):
        infer = self(batch)
        loss, acc = self.get_loss(batch, infer), self.get_acc(batch, infer)
        logs = {'loss': loss, 'acc': acc}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, index):
        infer = self(batch)
        loss, acc = self.get_loss(batch, infer), self.get_acc(batch, infer)
        return {'val_loss': loss, 'val_acc': acc}

    def test_step(self, batch, index):
        infer = self(batch)
        loss, acc = self.get_loss(batch, infer), self.get_acc(batch, infer)
        return {'test_loss': loss, 'test_acc': acc}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        val_acc_mean = torch.stack([x['val_acc'] for x in outputs]).mean()
        return {'log': {'val_loss': val_loss_mean, 'val_acc': val_acc_mean}}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        test_acc_mean = torch.stack([x['test_acc'] for x in outputs]).mean()
        return {'log': {'test_loss': test_loss_mean, 'test_acc': test_acc_mean}}

    def train_dataloader(self):
        dataset = KSXDataset(self.hp, mode='train')
        loader = DataLoader(dataset, batch_size=self.hp.train.batch_size, shuffle=True, num_workers=64)
        return loader

    def val_dataloader(self):
        dataset = KSXDataset(self.hp, mode='valid')
        loader = DataLoader(dataset, batch_size=self.hp.train.batch_size, num_workers=64)
        return loader

    def test_dataloader(self):
        dataset = KSXDataset(self.hp, mode='tests')
        loader = DataLoader(dataset, batch_size=self.hp.train.batch_size, num_workers=64)
        return loader
