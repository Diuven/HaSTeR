import argparse
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import Callback
from model import SimpleCNN, SATNet, model_list
from omegaconf import OmegaConf
from dataset import DemoDataset
from torch.utils.data import DataLoader

from utils.hangulUtils import HangulUtil


class TestCallback(Callback):
    def on_test_end(self, trainer, pl_module):
        targ_list = pl_module.test_res_targ
        pred_list = pl_module.test_res_pred
        for i in range(len(targ_list)):
            targ = HangulUtil.caption_to_jamos(targ_list[i][0])
            pred = HangulUtil.caption_to_jamos(pred_list[i][0])
            print("Target: %7s, Result: %7s" % (targ, pred))
        print("Done!!")
    

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
        logger=logger,
        callbacks=[TestCallback()]
    )

    trainer.test(model, test_dataloaders=loader)
    print("Done!")


if __name__ == '__main__':
    main()
