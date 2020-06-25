import argparse
from pytorch_lightning import Trainer, loggers
from torch.utils.data import DataLoader
from dataset import KSXAug
from model import SimpleCNN
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-chkp', type=str, help='Location of the checkpoint of the model')
    parser.add_argument('--cpu', action='store_true', help='Use cpu only')
    args = parser.parse_args()

    hp = OmegaConf.load('config/default.yaml')

    train_set = KSXAug(hp=hp, mode='train')
    train_loader = DataLoader(train_set, batch_size=hp.train.batch_size, shuffle=True, num_workers=16)

    train_name = 'SimpleCNN_%s' % hp.data.name
    logger = loggers.TensorBoardLogger('logs/', name=train_name)
    logger.log_hyperparams(OmegaConf.to_container(hp))

    trainer = Trainer(
        gpus=None if args.cpu else -1,
        logger=logger,
        resume_from_checkpoint=args.checkpoint
    )
    model = SimpleCNN(hp=hp)
    trainer.fit(model, train_dataloader=train_loader)


if __name__ == '__main__':
    main()
