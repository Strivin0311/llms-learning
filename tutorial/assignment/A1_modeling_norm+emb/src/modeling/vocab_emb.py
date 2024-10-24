from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup


class ParallelVocabEmbedding(nn.Module):
    """Parallel Vocab Embedding module
    This is a simplified version of the practical one, \
        which shards the vocabulary embedding table into `world_size` partitions in a process group, and \
        each rank in that process group only handles one partition of it, thus \
        in pratical, we can apply the large vocabulary embedding in parallel and then reduce them together.
    However, for this simpler module, you only need to implement the jobs for any single rank, and \
        don't have to handle the reduce operation or any other parallelism stuff.
    """
    
    def __init__(self, 
        vocab_size: int, 
        emb_size: int,
        rank: int = 0,
        world_size: int = 1,
        process_group: Optional[ProcessGroup] = None,
        init_mean: float = 0.0,
        init_std: float = 1.0,
        init_base_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ) -> None:
        """Initialize Parallel Vocab Embedding module
        
        Args:
            vocab_size(int): vocabulary size
            emb_size(int): embedding size
            rank(int, default = 0): rank
            world_size(int, default = 1): world size
            process_group(Optional[ProcessGroup], default = None): the process group (which will not be used for this simpler module yet)
            init_mean(float, default = 0.0): mean of the normal distribution
            init_std(float, default = 1.0): standard deviation of the normal distribution
            init_base_seed(int, default = 42): the base seed for the initialization (the real seed will be base seed + rank)
            dtype(torch.dtype, default = torch.float32): parameter dtype
            device(str, default = "cpu"): parameter device
        """
        super().__init__()
        raise NotImplementedError("TODO: Assignment1 - Task2")
        
    def forward(self, input_ids: torch.LongTensor) -> torch.Tensor:
        """The forward pass for Parallel Vocab Embedding module
        
        Args:
            input_ids(torch.LongTensor): input ids, with shape: (batch_size, seq_len)
        
        Returns:
            output(torch.Tensor): output embedding tensor, with shape: (batch_size, seq_len, emb_size)
        """
        raise NotImplementedError("TODO: Assignment1 - Task2")
        
    def reset_parameters(self) -> None:
        """Initialize learnable embedding parameters for Vocab Embedding from a normal distribution"""
        raise NotImplementedError("TODO: Assignment1 - Task2")
        