import torch
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam
from pytorch_lightning.core.lightning import LightningModule


class SimpleCNN(LightningModule):
    def __init__(self, hp):
        super(SimpleCNN, self).__init__()

        self.hp = hp
        self.height, self.width = hp.data.height, hp.data.width
        self.flat_size = self.height * self.width // 32 // 32 * 256
        self.class_count = hp.data.class_count

        self.feature = nn.Sequential(
            nn.Conv2d(3, 32, 7, padding=3), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 32, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 5, padding=2), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.classify = nn.Sequential(
            nn.Linear(self.flat_size, 1024), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, self.hp.data.class_count), nn.Softmax(dim=1)
        )

    def forward(self, x):
        feat = self.feature(x)
        flat = torch.flatten(feat, start_dim=1)
        pred = self.classify(flat)
        return pred

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=1e-4)

    def training_step(self, batch, index):
        img, lab = batch
        pred = self(img)
        loss = F.cross_entropy(pred, lab)
        logs = {'loss': loss}
        return {'loss': loss, 'log': logs}
