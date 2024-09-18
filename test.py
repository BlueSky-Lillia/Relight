import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer
import torch
from torch.utils.data import DataLoader, Dataset

class SimpleDataset(Dataset):
    def __init__(self):
        self.data = torch.randn(100, 1)
        self.targets = torch.randint(0, 2, (100,))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.targets[idx]

class SimpleModel(LightningModule):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(1, 2)

    def forward(self, x):
        return self.layer(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = torch.nn.functional.cross_entropy(y_hat, y)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-3)

dataset = SimpleDataset()
dataloader = DataLoader(dataset, batch_size=32)
model = SimpleModel()

trainer = pl.Trainer(
    accelerator='gpu',
    devices=[0, 1],
    precision=32
)

trainer.fit(model, dataloader)
