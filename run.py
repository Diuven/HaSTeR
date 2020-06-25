import argparse
from pytorch_lightning import Trainer, loggers
from torch.utils.data import DataLoader
from dataset import KSXAug
from model import SimpleCNN
from omegaconf import OmegaConf


def main():
    # parser = argparse.ArgumentParser()
    # args = parser.parse_args()

    hp = OmegaConf.load('config/default.yaml')

    train_set = KSXAug(hp=hp, mode='train')
    train_loader = DataLoader(train_set, batch_size=hp.train.batch_size, shuffle=True, num_workers=16)

    logger = loggers.TensorBoardLogger('logs/')

    trainer = Trainer(gpus=1, logger=logger)
    model = SimpleCNN(hp=hp)
    trainer.fit(model, train_dataloader=train_loader)


if __name__ == '__main__':
    main()
