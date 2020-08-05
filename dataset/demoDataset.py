import torch
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF
import os
from glob import glob
from PIL import Image
from pathlib import Path
from random import randrange, random
from utils.hangulUtils import KSXUtil, HangulUtil


class DemoDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.dataset_files = sorted(
            glob(os.path.join(self.data_dir, '*.jpg')),
            key=lambda name: int(Path(name).name.split("_")[0])
        )

    def __len__(self):
        return len(self.dataset_files)

    def __getitem__(self, idx):
        name = self.dataset_files[idx]
        img = Image.open(name)

        # if self.hp.data.grayscale:
        #     img = img.convert("L")

        arr = TF.to_tensor(img)
        # if self.hp.data.normalize:
        #     arr = TF.normalize(arr, 0, 1)

        lab = Path(name).stem.split("_")[1]

        # if self.hp.data.combined:
        #     targ = KSXUtil.ksx_index(lab)
        # else:
        #     targ = HangulUtil.letter_to_caption(lab)
        #     targ = torch.LongTensor(targ)
        targ = HangulUtil.letter_to_caption(lab)
        targ = torch.LongTensor(targ)

        return arr, targ
