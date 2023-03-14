import pytorch_lightning as pl
from torch.utils.data import DataLoader
from structure.pytorch.data.tiny_imagenet import TinyImageNet


class TinyImageNetDataModule(pl.LightningDataModule):
    def __init__(self, batch_size:int, root: str, num_workers: int=2, pin_memory: bool=True, train_transforms=None, val_transforms=None):
        super().__init__()
        self.batch_size = batch_size
        self.root = root
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_transforms, self.val_transforms = train_transforms, val_transforms
        self.setup()

    def setup(self, stage=None):
        self.train_data = TinyImageNet(root=self.root, split='train', transform=self.train_transforms)
        self.val_data = TinyImageNet(root=self.root, split='val', transform=self.val_transforms)

    def train_dataloader(self):
        return DataLoader(
            self.train_data,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )