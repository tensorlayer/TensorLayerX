#! /usr/bin/python
# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from tensorlayerx.backend import BACKEND

class DataLoader:
    def __new__(
        cls,
        dataset,
        batch_size=1,
        shuffle=False,
        drop_last=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=None,
        time_out=0,
        worker_init_fn=None,
        prefetch_factor=2,
        persistent_workers=False,
        ):
        if BACKEND == 'paddle':
            from paddle.io import DataLoader
            return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn, timeout=time_out, worker_init_fn=worker_init_fn, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers)
        else:
            from .dataloader import DataLoader
            return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last, sampler=sampler, batch_sampler=batch_sampler, num_workers=num_workers, collate_fn=collate_fn, persistent_workers=persistent_workers)
from .sampler import *
from .dataset import *