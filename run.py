import argparse
from pytorch_lightning import Trainer, loggers
from torch.utils.data import DataLoader
from dataset import KSXDataset
from model import SimpleCNN
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', '-chkp', type=str, help='Checkpoint of the model to use')
    parser.add_argument('--cpu', action='store_true', help='Use cpu only')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Config file to use')
    args = parser.parse_args()

    hp = OmegaConf.load(args.config)

    train_name = 'SimpleCNN_%s' % hp.data.name
    logger = loggers.TensorBoardLogger('logs/', name=train_name)
    logger.log_hyperparams(OmegaConf.to_container(hp))

    model = SimpleCNN(hp=hp)

    trainer = Trainer(
        gpus=None if args.cpu else -1,
        logger=logger,
        resume_from_checkpoint=args.checkpoint,
        max_epochs=100000,
        check_val_every_n_epoch=5
    )
    trainer.fit(model)
    trainer.test(model)


if __name__ == '__main__':
    main()
