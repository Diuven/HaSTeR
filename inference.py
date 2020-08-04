import argparse
from pytorch_lightning import Trainer, loggers
from model import SimpleCNN, SATNet, model_list
from omegaconf import OmegaConf
from dataset import DemoDataset
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('datapath', type=str, help='Path to files to infer')
    parser.add_argument('--checkpoint', '-chkp', type=str, required=True, help='Checkpoint of the model to use')
    parser.add_argument('--model', type=str, default='SATNet', help='Path to files to infer')
    parser.add_argument('--cpu', action='store_true', help='Use cpu for inference')
    args = parser.parse_args()

    if args.model == 'SimpleCNN':
        model = SimpleCNN.load_from_checkpoint(args.checkpoint)
    elif args.model == 'SATNet':
        model = SATNet.load_from_checkpoint(args.checkpoint)
    else:
        raise RuntimeError("Wrong model name %s" % args.model)
    model.freeze()

    train_name = 'infer_%s' % args.model
    logger = loggers.TensorBoardLogger('logs/', name=train_name)

    dataset = DemoDataset(args.datapath)
    loader = DataLoader(dataset, num_workers=64)

    trainer = Trainer(
        gpus=None if args.cpu else -1,
        logger=logger
    )

    trainer.test(model, test_dataloaders=loader)
    print("Done!")


if __name__ == '__main__':
    main()
