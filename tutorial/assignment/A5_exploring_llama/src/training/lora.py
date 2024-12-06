import os
import re

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modeling.models.base import BaseTokenizer, BaseModel

from ..modeling.datasets.base import BaseDataset

from ..modeling.config import (
    config_dataclass,
    make_required_field,
    make_fixed_field,
)

from ..utils import save_safetensors

from .base import BaseTrainConfig, BaseTrainer


@config_dataclass
class LoRATrainConfig(BaseTrainConfig):
    """LoRA Training Configurations Dataclass"""
    
    lora_weight_A_pattern: str = make_required_field()
    lora_weight_B_pattern: str = make_required_field()
    
    save_only_lora: bool = False
    
    
class LoRATrainer(BaseTrainer):
    """LoRA Trainer module
    Based the common APIs provided by `BaseTrainer`, \
        overwrite some of them to support LoRA fine-tuning
    """
    
    def __init__(
        self,
        config: LoRATrainConfig,
        model: BaseModel,
        tokenizer: BaseTokenizer,
        train_dataset: BaseDataset,
        eval_dataset: BaseDataset,
    ):
        """Initialize LoRA Trainer module
        
        Args:
            config (LoRATrainConfig): LoRA training configurations
            model (BaseModel): Base model
            tokenizer (BaseTokenizer): Base tokenizer
            train_dataset (BaseDataset): Training dataset
            eval_dataset (Optional[BaseDataset], optional): Evaluation dataset. Defaults to None.
        """
        super().__init__(
            config=config,
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )
        raise NotImplementedError("TODO: Assignment5 - Task3")
            
    def _save_ckpt(self, step: int) -> None:
        """Save the model as a checkpoint, \
            called in `self.run()` when the saving criterion is met
            
        NOTE: as for LoRA, we can choose to only save the LoRA adapter parameters, \
            since the parameters of the base model are not trainable and can be load from another checkpoint directory individually
        
        Args:
            step (int): current training step
        """
        raise NotImplementedError("TODO: Assignment5 - Task3")
        
       
        
        
    