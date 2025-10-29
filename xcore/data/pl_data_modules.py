from functools import partial
from typing import Any, Union, List, Optional, Sequence

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader, Dataset


class CrossDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        batch_sizes: DictConfig,
        num_workers: DictConfig,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_sizes = batch_sizes
        self.reload_dataloaders_every_n_epochs = 1

        # data
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.val_dataset = hydra.utils.instantiate(self.dataset.val)
            self.train_dataset = hydra.utils.instantiate(self.dataset.train)
        if stage == "test" or stage is None:
            self.test_dataset = hydra.utils.instantiate(self.dataset.test)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:

        return DataLoader(
            self.train_dataset,
            shuffle=False,
            batch_size=self.batch_sizes.train,
            num_workers=self.num_workers.train,
            collate_fn=self.train_dataset.collate_fn,
        )

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_sizes.val,
            num_workers=self.num_workers.val,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_sizes.test,
            num_workers=self.num_workers.test,
            collate_fn=self.test_dataset.collate_fn,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.dataset=}, " f"{self.num_workers=}, " f"{self.batch_sizes=})"


class SequentialLoader:
    def __init__(self, *dataloaders: DataLoader):
        self.dataloaders = dataloaders

    def __len__(self):
        return sum(len(d) for d in self.dataloaders)

    def __iter__(self):
        for dataloader in self.dataloaders:
            yield from dataloader


class JointDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset: DictConfig,
        batch_sizes: DictConfig,
        num_workers: DictConfig,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.dataset = dataset
        self.num_workers = num_workers
        self.batch_sizes = batch_sizes

        # data
        self.train_dataset: Optional[Dataset] = None
        self.val_dataset: Optional[Dataset] = None
        self.test_dataset: Optional[Dataset] = None

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            self.val_dataset = hydra.utils.instantiate(self.dataset.val)
            self.train_onto = hydra.utils.instantiate(self.dataset.train_onto)
            self.train_preco = hydra.utils.instantiate(self.dataset.train_preco)
            self.train_litbank = hydra.utils.instantiate(self.dataset.train_litbank)
        if stage == "test" or stage is None:
            self.test_dataset = hydra.utils.instantiate(self.dataset.test)

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        return SequentialLoader(
            *[
                DataLoader(
                    self.train_onto,
                    batch_size=self.batch_sizes.train,
                    num_workers=self.num_workers.train,
                    sampler=torch.utils.data.RandomSampler(self.train_onto, num_samples=1001),
                    collate_fn=self.train_onto.collate_fn,
                ),
                DataLoader(
                    self.train_preco,
                    batch_size=self.batch_sizes.train,
                    num_workers=self.num_workers.train,
                    sampler=torch.utils.data.RandomSampler(self.train_preco, num_samples=1001),
                    collate_fn=self.train_preco.collate_fn,
                ),
                DataLoader(
                    self.train_litbank,
                    shuffle=True,
                    batch_size=self.batch_sizes.train,
                    num_workers=self.num_workers.train,
                    collate_fn=self.train_litbank.collate_fn,
                ),
            ]
        )

    def val_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.batch_sizes.val,
            num_workers=self.num_workers.val,
            collate_fn=self.val_dataset.collate_fn,
        )

    def test_dataloader(self, *args, **kwargs) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.batch_sizes.test,
            num_workers=self.num_workers.test,
            collate_fn=self.test_dataset.collate_fn,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(" f"{self.dataset=}, " f"{self.num_workers=}, " f"{self.batch_sizes=})"
