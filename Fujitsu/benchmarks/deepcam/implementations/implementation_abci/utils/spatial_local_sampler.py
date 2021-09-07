import math
import random
from typing import TypeVar, Optional, Iterator

import torch
from torch.utils.data import Sampler, Dataset
import torch.distributed as dist
#from . import RndnumBoxMuller


T_co = TypeVar('T_co', covariant=True)


class SpatialLocalSampler(Sampler[T_co]):
    r"""Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class:`~torch.utils.data.LocalSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Args:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`world_size` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.
        drop_last (bool, optional): if ``True``, then the sampler will drop the
            tail of the data to make it evenly divisible across the number of
            replicas. If ``False``, the sampler will add extra indices to make
            the data evenly divisible across the replicas. Default: ``False``.
        global_fraction(float, optional): in range of [0,1]. If ratio = 0, 
            it similar to local sampler. If ratio = 1, it is similar  
            to default distributed global sampler. Default value = 1
        
    .. warning::
        In distributed mode, calling the :meth:`set_epoch` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = LocalSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, global_fraction:float = 1.0, shuffle: bool = True,
                 seed: int = 0, drop_last: bool = False) -> None:      
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))

        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        if (global_fraction > 1) or (global_fraction < 0): 
            raise ValueError(
                "Invalid global_fraction {}, global_fraction should be in the interval [0, 1]")
        self.global_fraction = global_fraction
        # If the dataset length is evenly divisible by # of replicas, then there
        # is no need to drop any data, since the dataset will be split equally.
        if self.drop_last and len(self.dataset) % self.num_replicas != 0:  # type: ignore
            # Split to nearest available length that is evenly divisible.
            # This is to ensure each rank receives the same amount of data when
            # using this Sampler.
            self.num_samples = math.ceil(
                # `type:ignore` is required because Dataset cannot provide a default __len__
                # see NOTE in pytorch/torch/utils/data/sampler.py
                (len(self.dataset) - self.num_replicas) / self.num_replicas  # type: ignore
            )
        else:
            self.num_samples = math.ceil(len(self.dataset) / self.num_replicas)  # type: ignore
        self.total_size = self.num_samples * self.num_replicas
        self.shuffle = shuffle
        self.seed = seed

        ## For DEBUG
        # self.total_size = 48 #48
        # self.num_samples = 6 #12
        # self.len_dataset = 48 # 47
        self.len_dataset = len(self.dataset)
        
        #NOTE: Dataset keep the global indices so that the len of dataset is = total len of the sub actual datasets 
        
        # Divided indices into 2 part. (1) for distributed sampling, (2) for local sampling.
        # Those two partition then will be divided equally for all the ranks.
        # drop_last is used in the distributed sampling partitions.
        self.num_samples_global = int(self.global_fraction * self.num_samples)
        self.num_samples_local = self.num_samples - self.num_samples_global
        assert self.num_samples_local <= self.num_samples
        assert self.num_samples_global <= self.num_samples
        assert self.num_samples_local >= 0
        assert self.num_samples_global >= 0
        
        self.len_dataset_local = self.num_samples_local * self.num_replicas
        self.len_dataset_global = self.len_dataset - self.len_dataset_local
        self.total_size_local = self.len_dataset_local
        self.total_size_global = self.num_samples_global * self.num_replicas
        self.indices = None#Indices of each rank used in this epoch
        self.next_indices_local = [None] * self.num_replicas #Indices of each rank used in next epoch (used by the dataset-scheduler to move samples between ranks)
        self.next_indices_global = [None] * self.num_replicas
        
        #print("PartialLocalSampler: ",self.rank, self.len_dataset, self.total_size, self.num_samples, self.num_samples_local, self.num_samples_global, self.num_replicas, self.len_dataset_local, self.len_dataset_global, self.total_size_global)

    def _init_first_indices_list(self):
        self.indices = [None] * self.num_replicas
        if self.shuffle: # Shuffle locally
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)

            for i in range(0, self.num_replicas):
                max_index = self.num_samples
                if i == self.num_replicas - 1: # last rank.
                    if self.len_dataset < self.total_size:
                        max_index = self.len_dataset - self.num_samples * (self.num_replicas - 1)
                
                indices_rank = torch.randperm(max_index, generator=g)
                indices_rank = indices_rank + self.num_samples * i
                self.indices[i] = indices_rank.tolist()
        else:
            for i in range(0, self.num_replicas):
                max_index = self.num_samples
                if i == self.num_replicas - 1: # last rank.
                    if len_dataset < self.total_size:
                        max_index = len_dataset - self.num_samples * (self.num_replicas - 1)
                self.indices[i] = list(range(self.num_samples * i, max_index))

        
        if not self.drop_last:
            # add extra samples to make it evenly divided
            padding_size = self.num_samples - len(self.indices[self.num_replicas - 1])
            if padding_size <= len(self.indices):
                self.indices[self.num_replicas - 1] += self.indices[self.num_replicas - 1][:padding_size]
            else:
                self.indices[self.num_replicas - 1] += (self.indices[self.num_replicas - 1] * math.ceil(padding_size / len(self.indices[self.num_replicas - 1])))[:padding_size]
    
    def select_global_indices(self,list_indices,rank_i):
        return list_indices[rank_i][self.num_samples_local:self.num_samples]
    
    def select_local_indices(self,list_indices,rank_i):
        return list_indices[rank_i][0:self.num_samples_local]
    
    def generate_next_indices_list(self) -> None:
        random.seed(self.seed + self.epoch)
        #print(self.rank, self.indices)
        indices_global_all = []
        for i in range(0, self.num_replicas):
            indices_global_all = indices_global_all + self.select_global_indices(self.indices,i)
        if self.shuffle:   
            random.shuffle(indices_global_all)        

        #self.indices = [None] * self.num_replicas
        for i in range(0, self.num_replicas):
            self.next_indices_local[i] = self.select_local_indices(self.indices,i)
            self.next_indices_global[i] = indices_global_all[i:self.total_size_global:self.num_replicas]
    
    def generate_this_indices_list(self) -> None:
        # Generate from next_indices_list
        random.seed(self.seed + self.epoch)
        self.indices = [None] * self.num_replicas    
        for i in range(0, self.num_replicas):
            self.indices[i] = self.next_indices_local[i] + self.next_indices_global[i]
            if self.shuffle:   
                random.shuffle(self.indices[i])            
            assert len(self.indices[i]) == self.num_samples
            
    def next_epoch(self):
        if self.indices is None:
            self._init_first_indices_list()
        else:
            self.generate_this_indices_list()
            
        self.generate_next_indices_list()
        #print(self.rank, self.next_indices_global)
  
    def get_this_global_indices_list(self) -> None:
        this_indices_global = [None] * self.num_replicas
        for i in range(0, self.num_replicas):
            this_indices_global[i] = self.select_global_indices(self.indices,i)
        return this_indices_global
    
    def get_next_global_indices_list(self) -> None:
        return self.next_indices_global
  
    def __iter__(self) -> Iterator[T_co]:
        # if self.rank ==0:
            # print(self.indices[0])
        if self.indices is None:
            self.generate_this_indices_list()
            self.generate_next_indices_list()
        
        return iter(self.indices[self.rank])   

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Args:
            epoch (int): Epoch number.
        """
        self.epoch = epoch
        self.next_epoch()
    
