# -*- coding: utf-8 -*-

import torch
from torch.utils.data import DataLoader


def get_dataloader(dataset: torch.utils.data.Dataset,
                   batch_size: int,
                   shuffle: bool = True,
                   num_workers: int = 0,
                   drop_last: bool = False,
                   pin_memory: bool = False,
                   balance: bool = False,
                   ):
    """Return a `DataLoader` instance."""
    if balance:
        from datasets.samplers import ImbalancedDatasetSampler  # pylint: disable=import-outside-toplevel
        sampler = ImbalancedDatasetSampler(dataset)
    else:
        sampler = None

    loader_configs = {
        'dataset': dataset,
        'batch_size': batch_size,
        'shuffle': shuffle,
        'sampler': sampler,
        'num_workers': num_workers,
        'drop_last': drop_last,
        'pin_memory': pin_memory,
    }
    return DataLoader(**loader_configs)
