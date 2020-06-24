from torch.utils.data import Dataset
from torchvision import transforms
import os
from glob import glob
from PIL import Image
from pathlib import Path
from random import randrange, random
from utils.hangulUtils import KSXUtil


def augment_image(img):
    pix = img.load()

    for _ in range(100):
        i, j = randrange(0, img.size[0]), randrange(0, img.size[1])
        pix[i,j] = (0,0,0) if random() < 0.5 else (255,255,255)

    img = transforms.ColorJitter(0.7, 0.7, 0.5, 0.3)(img)

    return img


class KSXAug(Dataset):
    def __init__(self, hp, mode):
        self.hp = hp
        self.mode = mode

        if mode is 'train':
            self.data_dir = self.hp.data.train_dir
        elif mode is 'test':
            self.data_dir = self.hp.data.test_dir
        else:
            raise ValueError(f"invalid dataloader mode {mode}")

        self.dataset_files = sorted(
            map(
                os.path.abspath,
                glob(os.path.join(self.data_dir, self.hp.data.file_format)),
            )
        )

    def __len__(self):
        return len(self.dataset_files)

    def __getitem__(self, idx):
        name = self.dataset_files[idx]
        img = Image.open(name)

        if self.mode is 'train':
            img = augment_image(img)
        arr = transforms.ToTensor()(img)

        lab = Path(name).stem.split("_")[1]

        return arr, KSXUtil.ksx_index(lab)
