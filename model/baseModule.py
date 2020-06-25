import torch
from torch.utils.data import DataLoader
from pytorch_lightning.core.lightning import LightningModule
from abc import ABC, abstractmethod
from dataset import KSXDataset


class BaseModule(LightningModule, ABC):
    def __init__(self, hp):
        super().__init__()
        self.hp = hp

    @abstractmethod
    def get_loss(self, batch):
        pass

    def training_step(self, batch, index):
        loss = self.get_loss(batch)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}

    def validation_step(self, batch, index):
        loss = self.get_loss(batch)
        return {'val_loss': loss}

    def test_step(self, batch, index):
        loss = self.get_loss(batch)
        return {'test_loss': loss}

    def validation_epoch_end(self, outputs):
        val_loss_mean = torch.stack([x['val_loss'] for x in outputs]).mean()
        return {'log': {'val_loss': val_loss_mean}}

    def test_epoch_end(self, outputs):
        test_loss_mean = torch.stack([x['test_loss'] for x in outputs]).mean()
        return {'log': {'test_loss': test_loss_mean}}

    def train_dataloader(self):
        dataset = KSXDataset(self.hp, mode='train')
        loader = DataLoader(dataset, batch_size=self.hp.train.batch_size, shuffle=True, num_workers=16)
        return loader

    def val_dataloader(self):
        dataset = KSXDataset(self.hp, mode='valid')
        loader = DataLoader(dataset, batch_size=self.hp.train.batch_size, num_workers=16)
        return loader

    def test_dataloader(self):
        dataset = KSXDataset(self.hp, mode='tests')
        loader = DataLoader(dataset, batch_size=self.hp.train.batch_size, num_workers=16)
        return loader
