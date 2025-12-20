import numpy as np
import random
from monai.data import list_data_collate, decollate_batch, DataLoader, ThreadDataLoader
from torch.utils.data.dataloader import default_collate
from functools import partial
from torch.utils.data.sampler import SubsetRandomSampler, WeightedRandomSampler
import torch


def _base_worker_init_fn(worker_id: int, seed: int):
    """
    Module‐level worker_init_fn so it can be pickled.
    """
    np.random.seed(seed + worker_id)
    random.seed(seed + worker_id)


class BaseDataLoader(torch.utils.data.DataLoader):
    """
    Base class for all data loaders
    """

    def __init__(self,
                dataset,
                batch_size,
                shuffle,
                validation_split,
                num_workers,
                seed: int = 0,
                pin_memory: bool = True,
                persistent_workers: bool = None,
                drop_last: bool = True,
                collate_fn=default_collate,):
        self.validation_split = validation_split
        self.shuffle = shuffle
        self.seed = seed

        self.batch_idx = 0
        self.n_samples = len(dataset)
        if persistent_workers is None:
            persistent_workers = (num_workers > 0)

        self.sampler, self.valid_sampler = self._split_sampler(self.validation_split)
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        worker_init = partial(_base_worker_init_fn, seed=self.seed)

        self.init_kwargs = {
            'dataset': dataset,
            'batch_size': batch_size,
            'shuffle': self.shuffle,
            'collate_fn': collate_fn,
            'num_workers': num_workers,
            'persistent_workers': persistent_workers,
            'worker_init_fn': worker_init
        }
        super().__init__(sampler=self.sampler, pin_memory=self.pin_memory, drop_last=self.drop_last, **self.init_kwargs)

    def _split_sampler(self, split):
        if split == 0.0:
            return None, None
        idx_full = np.arange(self.n_samples)

        np.random.seed(0)
        np.random.shuffle(idx_full)

        if isinstance(split, int):
            assert split > 0
            assert split < self.n_samples, "validation set size is configured to be larger than entire dataset."
            len_valid = split
        else:
            len_valid = int(self.n_samples * split)

        valid_idx = idx_full[0:len_valid]
        train_idx = np.delete(idx_full, np.arange(0, len_valid))

        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        # turn off shuffle option which is mutually exclusive with sampler
        self.shuffle = False
        self.n_samples = len(train_idx)

        return train_sampler, valid_sampler

    def split_validation(self):
        if self.valid_sampler is None:
            return None
        else:
            return ThreadDataLoader(sampler=self.valid_sampler, pin_memory=self.pin_memory, drop_last=self.drop_last,
                                    **self.init_kwargs)
    
    
