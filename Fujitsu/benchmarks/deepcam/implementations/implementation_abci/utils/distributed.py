# Ported from https://github.com/pytorch/pytorch/blob/v1.6.0/torch/utils/data/distributed.py
# Modified by Fujitsu

import math
import torch
import random
#from . import Sampler
from torch.utils.data import Sampler
import torch.distributed as dist
import multiprocessing
import pandas as pd
from parse import parse


class DistributedSampler(Sampler):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, seed=0, dataset_size=-1, comm_size=1, comm_rank=0, importance_sampling_mode="disabled"):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.comm_rank = comm_rank
        self.comm_size = comm_size
        self.epoch = 0
        if dataset_size > 0:
            self.num_samples = int(math.ceil(dataset_size * 1.0 / self.num_replicas))
        else:
            self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = int(seed)

        self.indices = None
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        for i in range(0, self.comm_size):
            indices_rank = torch.randperm(len(self.dataset), generator=g)
            if i == self.comm_rank:
                self.indices = indices_rank.tolist()

        self.importance_sampling_mode = importance_sampling_mode
        self.importance_scores = []
        self.importance_scores_lock = multiprocessing.Lock()
        # All samples are important in the first epoch
        self.important_samples = [True] * len(self.dataset)
        self.loss_df = None

    def append_importance(self, idx, importance):
        self.importance_scores_lock.acquire()
        self.importance_scores.append((importance, idx))
        self.importance_scores_lock.release()

    def is_sample_important(self, idx):
        return self.important_samples[idx]

    def __iter__(self):
        if self.comm_rank == 0:
            print("epoch: {}, importance_sampling_mode: {}".format(self.epoch, self.importance_sampling_mode))

        if (self.importance_sampling_mode != "disabled" and self.epoch > 0):
            self.importance_scores.sort(reverse=True)
            importance_tuple = tuple(map(list, zip(*self.importance_scores)))
            tmp_losses = importance_tuple[0]
            tmp_indices = importance_tuple[1]
            drop_ratio = 0.0

            self.important_samples = [False] * len(self.dataset)
            # Mark important samples based on drop-off percentage
            # and move unimportant ones to the front of the list
            indices = []

            # Figure out the ratio to drop
            for s in self.importance_sampling_mode.split("+"):
                r = parse("drop-{:d}perc", s)
                if r:
                    drop_ratio = (float(r[0]) / 100)

            if drop_ratio > 0.0:
                for i in range(len(tmp_indices)):
                    if i < (1.0 - drop_ratio) * len(self.dataset):
                        self.important_samples[tmp_indices[i]] = True
                        indices.append(tmp_indices[i])
                    else:
                        indices.insert(0, tmp_indices[i])

            # Baseline behavior (dummy) mode for loss logging, all samples important
            if self.importance_sampling_mode == "loss-logger":
                random.seed(self.seed + self.epoch)
                dummy_indices = list(range(len(self.dataset)))
                for i in range(0, self.comm_size):
                    if i == self.comm_rank:
                        # deterministically shuffle based on epoch and seed
                        random.shuffle(self.indices)
                    else:
                        random.shuffle(dummy_indices)
                indices = self.indices
                self.important_samples = [True] * len(self.dataset)
                filenames = [self.dataset.samples[k] for k in tmp_indices]
                epoch_loss_df = pd.DataFrame.from_dict({
                    "Rank": [self.comm_rank] * len(tmp_indices),
                    "Filename": filenames,
                    "Sample_ID": tmp_indices,
                    "Epoch": [self.epoch] * len(tmp_indices),
                    "Loss": tmp_losses})
                self.loss_df = pd.concat([self.loss_df, epoch_loss_df], ignore_index=True)
                if self.epoch >= 29:
                    self.loss_df.to_feather("losses-rank-%04d.feather" % self.comm_rank)


            self.indices = indices
            if self.comm_rank == 0:
                print("epoch: {}, drop_ratio: {}".format(self.epoch, drop_ratio))
                print("epoch: {}, importance_scores: {}".format(self.epoch, self.importance_scores))
                print("epoch: {}, indices: {}".format(self.epoch, self.indices))
            self.importance_scores = []


        elif self.shuffle and self.epoch > 0:
            random.seed(self.seed + self.epoch)
            tmp_indices = list(range(len(self.dataset)))
            for i in range(0, self.comm_size):
                if i == self.comm_rank:
                    # deterministically shuffle based on epoch and seed
                    random.shuffle(self.indices)
                else:
                    random.shuffle(tmp_indices)

        #if self.comm_rank == 0:
        #    print(self.indices)
        return iter(self.indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): Epoch number.
        """
        self.epoch = int(epoch)
