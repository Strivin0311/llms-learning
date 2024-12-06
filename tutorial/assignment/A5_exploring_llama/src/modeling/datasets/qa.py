from typing import Dict, Any, List, Union, Optional, Tuple, Sequence
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import (
    config_dataclass,
    make_required_field,
    make_fixed_field,
)

from ..models.base import BaseTokenizer

from ...utils import load_jsonl

from .base import (
    BatchLayout,
    PaddingSide,
    TruncateSide,
    BaseDatasetConfig,
    BaseDataset,
)


@config_dataclass
class QADatasetConfig(BaseDatasetConfig):
    """Dataset Configurations Dataclass for Question-Answering Tasks"""
    
    question_key: str = make_fixed_field("question")
    answer_key: str = make_fixed_field("answer")
    
    question_prefix: str = make_fixed_field("QUESTION")
    answer_prefix: str = make_fixed_field("ANSWER")
    

class QADataset(BaseDataset):
    """Dataset Class for Question-Answering Tasks"""
    
    def __init__(
        self,
        config: QADatasetConfig,
        tokenizer: BaseTokenizer,
        data_files: Union[str, List[str]],
    ):
        """Initialize QADataset module
        Args:
            config (QADatasetConfig): qa dataset configuration dataclass object
            tokenizer (BaseTokenizer): tokenizer module
            data_files (Union[str, List[str]]): path to the file(s) with the data in .jsonl format
        """
        super().__init__()
        raise NotImplementedError("TODO: Assignment5 - Task3")
    
    def num_samples(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        raise NotImplementedError("TODO: Assignment5 - Task3")
    
    def sample(self, idx: int) -> Dict[str, Any]:
        """
        Returns a sample from the dataset.
        """
        raise NotImplementedError("TODO: Assignment5 - Task3")
    
    def num_batchs(self) -> int:
        """
        Returns the number of batchs in the dataset.
        """
        raise NotImplementedError("TODO: Assignment5 - Task3")
    
    def batch(self, idx: int) -> Dict[str, Any]:
        """
        Returns a batch from the dataset.
        """
        raise NotImplementedError("TODO: Assignment5 - Task3")
    
    def shuffle(self, seed: Optional[int] = None) -> None:
        """Shuffle the dataset, including the samples and batches.
            
        Args:
            seed (Optional[int], optional): Random seed. Defaults to None to be un-deterministic.
        """
        raise NotImplementedError("TODO: Assignment5 - Task3")
    