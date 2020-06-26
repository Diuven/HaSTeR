import argparse
from pytorch_lightning import Trainer, loggers
from model import SimpleCNN, SATNet, model_list
from omegaconf import OmegaConf


def main():
    parser = argparse.ArgumentParser()
    # Make this case-insensitive?
    parser.add_argument('--model', type=str, required=True, choices=model_list, help='Model to use')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='Config file to use')
    parser.add_argument('--checkpoint', '-chkp', type=str, help='Checkpoint of the model to use')
    parser.add_argument('--cpu', action='store_true', help='Use cpu only (Overrides some hyperparameters)')
    args = parser.parse_args()

    hp = OmegaConf.load(args.config)
    if args.cpu:
        hp.train.batch_size = 16

    if args.model == 'SimpleCNN':
        model = SimpleCNN(hp=hp)
    elif args.model == 'SATNet':
        model = SATNet(hp=hp)
    else:
        raise RuntimeError("Wrong model name %s" % args.model)

    train_name = '%s_%s' % (args.model, hp.data.name)
    logger = loggers.TensorBoardLogger('logs/', name=train_name)
    logger.log_hyperparams(OmegaConf.to_container(hp))

    trainer = Trainer(
        gpus=None if args.cpu else -1,
        logger=logger,
        resume_from_checkpoint=args.checkpoint,
        max_epochs=100000,
    )
    trainer.fit(model)
    trainer.test(model)


if __name__ == '__main__':
    main()
