from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Iterator
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import (
    BaseConfig,
    config_dataclass,
    make_required_field,
    make_fixed_field,
)

from ..prompt import (
    BatchLayout,
    PaddingSide,
    TruncateSide,
    PromptTemplate
)


@config_dataclass
class BaseDatasetConfig(BaseConfig):
    """Dataset Basic Configurations Dataclass"""
    
    # shape configurations
    seq_len: int = make_required_field()
    batch_size: int = make_required_field()
    batch_layout: BatchLayout = make_fixed_field(BatchLayout.STACK) # NOTE: we only allow stacking for simplicity
    
    # transformation configurations
    padding_side: PaddingSide = PaddingSide.LEFT
    truncate_side: TruncateSide = TruncateSide.RIGHT
    drop_last_incomplete_batch: bool = True
    
    # constants configurations
    samples_key: str = "samples"
    input_ids_key: str = "input_ids"
    labels_key: str = "labels"
    cu_seqlens_key: str = "cu_seqlens"
    ignore_idx: int = -100
    prefix_template: PromptTemplate = PromptTemplate("[{prefix}]: ")
    sep_str: str = "\n"
    
    # common configurations
    device: str = "cpu"
    
    
class BaseDataset(nn.Module, ABC):
    """Base Dataset Class"""
    
    @abstractmethod
    def num_samples(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
    
    @abstractmethod
    def sample(self, idx: int) -> Dict[str, Any]:
        """
        Returns a sample from the dataset.
        """
    
    def samples(self) -> Iterator[Dict[str, Any]]:
        """Generator that yields samples one by one until all samples have been iterated.
        
        Yields:
            Dict[str, Any]: A single sample from the dataset.
        """
        for idx in range(self.num_samples()):
            yield self.sample(idx)
    
    @abstractmethod
    def num_batchs(self) -> int:
        """
        Returns the number of batchs in the dataset.
        """
        
    @abstractmethod
    def batch(self, idx: int) -> Dict[str, Any]:
        """
        Returns a batch from the dataset.
        """
        
    def batches(self) -> Iterator[Dict[str, Any]]:
        """Generator that yields batches one by one until all batches have been iterated.
        
        Yields:
            Dict[str, Any]: A single batch from the dataset.
        """
        for idx in range(self.num_batchs()):
            yield self.batch(idx)
        
    @abstractmethod
    def shuffle(self, seed: Optional[int] = None) -> None:
        """Shuffle the dataset, including the samples and batches.
            
        Args:
            seed (Optional[int], optional): Random seed. Defaults to None to be un-deterministic.
        """