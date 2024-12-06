from typing import Dict, Any, Optional, Tuple, Union, List
from enum import Enum
import os
from glob import glob
from datetime import datetime
from itertools import cycle

from rich import print as rprint
import wandb
from torch.utils.tensorboard import SummaryWriter

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..modeling.models.base import BaseTokenizer, BaseModel

from ..modeling.datasets.base import BaseDatasetConfig, BaseDataset

from ..modeling.config import (
    BaseConfig,
    config_dataclass,
    make_required_field,
    make_fixed_field,
    make_factory_field,
)

from ..utils import (
    convert_to_list,
    seconds_to_hms_str,
    format_rich_text,
    check_valid_path,
    load_safetensors,
    save_safetensors,
)


class OptimizerType(Enum):
    """Optimizer Types Enum"""
    
    SGD = "sgd"
    ADAM = "adam"
    ADAMW = "adamw"
    

class TrainLogType(Enum):
    """Training Log Types Enum"""
    
    TERMINAL = "terminal"
    TENSORBOARD = "tensorboard"
    WANDB = "wandb"


@config_dataclass
class BaseTrainConfig(BaseConfig):
    """Base Training Configurations Dataclass"""
    
    # training configurations
    train_steps: int = make_required_field()
    
    # evaluation configurations
    eval_interval: Optional[int] = None
    eval_steps: int = 0
    
    # transformer configurations
    shuffle: bool = False
    shuffle_seed: Optional[int] = None
    
    # optimizer configurations
    optimizer_type: OptimizerType = OptimizerType.ADAMW
    learning_rate: float = make_required_field()
    momentum: float = 0. # NOTE: only used for SGD
    betas: Tuple[float, float] = (0.9, 0.999) # NOTE: only used for ADAM & ADAMW
    weight_decay: float = 0. # NOTE: only used for ADAMW
    
    # checkpoint configurations
    load_ckpt_dirs: Optional[Union[str, List[str]]] = None
    load_ckpt_step: bool = True
    
    save_interval: Optional[int] = None
    save_last_step: bool = True
    save_ckpt_dir: str = "." # NOTE: will be created if not exists
    
    max_shard_size: int = 1024 # NOTE: in unit: MB
    step_idx_width: int = 5
    ckpt_step_prefix: str = make_fixed_field("step-")
    ckpt_file_ext: str = make_fixed_field("safetensors")
    
    # logging configurations
    log_interval: Optional[int] = None
    log_last_step: bool = True
    log_types: Tuple[TrainLogType] = (TrainLogType.TERMINAL,)
    log_kwargs: dict = make_factory_field(dict)
    
    # common configurations
    device: str = "cpu"


class BaseTrainer(nn.Module):
    """Base Trainer module
    Define some common APIs for LLM training, \
        and provide the default implementations for these APIs
    """
    
    def __init__(
        self,
        config: BaseTrainConfig,
        model: BaseModel,
        tokenizer: BaseTokenizer,
        train_dataset: BaseDataset,
        eval_dataset: Optional[BaseDataset] = None,
    ):
        """Initialize Base Trainer module
        
        Args:
            config (BaseTrainConfig): Base training configurations
            model (BaseModel): Base model
            tokenizer (BaseTokenizer): Base tokenizer
            train_dataset (BaseDataset): Training dataset
            eval_dataset (Optional[BaseDataset], optional): Evaluation dataset. Defaults to None.
        """
        super().__init__()
        raise NotImplementedError("TODO: Assignment5 - Task3")
    
    def run(self) -> None:
        """Run the whole training steps, \
            until the stopping criterion is met
        NOTE: this is an one-time API, and you have to re-initialize a new trainer if you need to rerun
        """
        raise NotImplementedError("TODO: Assignment5 - Task3")
    
    def _train_step(
        self,
        batch_data: Dict[str, Any],
    ) -> torch.Tensor:
        """One training step pass, \
            called in `self.run()`:
            
            1. feed a batch of data to the model to apply forward pass with gradient tracking enabled to get the training loss
            2. apply backward pass to compute the gradients
            3. let the optimizer update the model parameters with the gradients
            
        Args:
            batch_data (Dict[str, Any]): a batch of data as a string-key dictionary
            
        Returns:
            torch.Tensor: the training loss
        """
        raise NotImplementedError("TODO: Assignment5 - Task3")
    
    @torch.no_grad()
    def _eval_step(
        self,
        batch_data: Dict[str, Any],
    ) -> torch.Tensor:
        """One evaluation step pass, \
            called in `self.run()` when the evaluation criterion is met:
            
            1. feed a batch of data to the model to apply forward pass with gradient tracking disabled to get the evaluation loss
        
        Args:
            batch_data (Dict[str, Any]): a batch of data as a string-key dictionary
            
        Returns:
            torch.Tensor: the evaluation loss
        """
        raise NotImplementedError("TODO: Assignment5 - Task3")
    
    def _load_ckpt(self) -> None:
        """Load the model from the pretrained checkpoint directory (or directories) \
            to resume training if needed, called in `self.__init__()`
        
        NOTE: if multiple checkpoints are provided and the parameter keys are overlapped, \
                the later ones will overwrite the earlier ones
        """
        raise NotImplementedError("TODO: Assignment5 - Task3")
    
    def _save_ckpt(self, step: int) -> None:
        """Save the model at the current training step as a checkpoint, \
            called in `self.run()` when the saving criterion is met
        
        Args:
            step (int): current training step
        """
        raise NotImplementedError("TODO: Assignment5 - Task3")
    