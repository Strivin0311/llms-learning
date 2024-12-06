from typing import List, Dict, Tuple, Optional, Union
from contextlib import contextmanager
from enum import Enum
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..modeling.models.base import BaseTokenizer, BaseModel

from ..modeling.datasets.base import BatchLayout, PaddingSide, TruncateSide

from ..modeling.prompt import PromptType, PromptTemplate

from ..modeling.config import (
    BaseConfig,
    config_dataclass,
    make_required_field,
    make_fixed_field,
)

from ..utils import convert_to_list


class DecodeStrategy(Enum):
    """Decode Strategies Enum"""
    
    GREEDY = "greedy"
    SAMPLING = "sampling"


@config_dataclass
class InferenceConfig(BaseConfig):
    """Inference Configurations Dataclass"""
    
    # generation configurations
    decode_strategy: DecodeStrategy = DecodeStrategy.GREEDY
    temperature: float = 1.0
    max_new_tokens: int = make_required_field() # NOTE: we allow neither infinite generation nor early stopping for simplicity
    top_p: float = 1.0 # NOTE: only used when using sampling decode strategy
    top_k: int = 50 # NOTE: only used when using sampling decode strategy
    streaming: bool = False # NOTE: used when only one single user query is requested at a time, i.e. `inferred_batch_size == 1`
    sampling_seed: Optional[int] = None # NOTE: only used when using sampling decode strategy, if None then do not set seed
    
    # padding configurations
    batch_layout: BatchLayout = make_fixed_field(BatchLayout.STACK) # NOTE: we only allow stacking for simplicity
    padding_side: PaddingSide = PaddingSide.LEFT
    pad_to_multiple_of: int = 1
    
    # truncate configurations
    truncate_length: Optional[int] = None # NOTE: if None, then no truncation
    truncate_side: TruncateSide = TruncateSide.RIGHT
    
    # common configurations
    device: str = "cpu"

    def __post_init__(self):
        """Post-initialization method for InferenceConfig"""
        super().__post_init__()

        assert self.pad_to_multiple_of > 0 and (
            (self.pad_to_multiple_of & (self.pad_to_multiple_of - 1)) == 0
        ), "pad_to_multiple_of must be a power of 2"

        if self.truncate_length is not None and self.truncate_side == TruncateSide.MIDDLE:
            assert self.truncate_length % 2 == 0, "truncate_length must be even when truncate_side is MIDDLE"


class InferenceAgent(nn.Module):
    """Inference Agent module"""
    
    def __init__(
        self,
        config: InferenceConfig,
        model: BaseModel,
        tokenizer: BaseTokenizer,
    ):
        """Initialize Inference Agent module
        
        Args:
            config (InferenceConfig): Inference Configurations
            model (BaseModel): the inner causal language model, which supports the common APIs of `BaseModel`
            tokenizer (BaseTokenizer): the inner tokenizer, which supports the common APIs of `BaseTokenizer`
        """
        super().__init__()
        raise NotImplementedError("TODO: Assignment5 - Task2")
        
    def set_prompt(
        self,
        prompt_template: PromptTemplate,
        prompt_type: PromptType = PromptType.SYSTEM,
    ) -> None:
        """Set the prompt template
        
        Args:
            prompt_template (PromptTemplate): the prompt template
            prompt_type (PromptType): the prompt type
        """
        raise NotImplementedError("TODO: Assignment5 - Task2")
            
    def get_prompt(
        self,
        prompt_type: PromptType = PromptType.SYSTEM
    ) -> PromptTemplate:
        """Get the prompt template
        
        Args:
            prompt_type (PromptType): the prompt type
        
        Returns:
            PromptTemplate: the prompt template
        """
        raise NotImplementedError("TODO: Assignment5 - Task2")
    
    @torch.no_grad()
    def forward(
        self, 
        query: Union[str, List[str]], 
        **kwargs: Optional[Dict[str, str]]
    ) -> List[Dict[PromptType, str]]:
        """The forward pass of the Inference Agent module
        
        Args:
            query (Union[str, List[str]]): a single query prompt or a batch of user query prompts \
                as the core distinct instructions to ask the model to respond, \
                appended to the end of the complete prompt with the same system prompt and context prompt
                NOTE: when is a streaming mode, the query should be a single prompt
            kwargs (dict): additional keyword arguments to be passed to format the prefixed prompt templates
                NOTE: if certain key in `kwargs` are found in both system prompt template and context prompt template, \
                    the corresponding value will share in both of them as well
        Returns:
            List[Dict[PromptType, str]]: the list of dictionaries, \
                each of which should contain every prompt type in `PromptType` (key) and the corresponding prompt (value)
            NOTE: to simplify, we do not use early stopping strategy since the stopping point for each response might vary, \
                thus the length of the latent token ids for each response is ensured to be `max_new_tokens`
        """
        raise NotImplementedError("TODO: Assignment5 - Task2")
    
    @staticmethod
    def load_generation_config(
        config_file: str, 
        **extra_configs
    ) -> InferenceConfig:
        """Load config from the original original Llama generation config
        
        Args:
            config_file(str): path to the config file of the original original Llama generation config in .json format
            extra_configs(dict, optional): extra (key, value) config pair(s), to overwrite `config.key = value`, \
                helpful to set some configurations that are neither fixed nor provided in the original config such as `device`, `seed`, etc.
                NOTE: if any required configuration is not found in the original config, you are supposed to pass it in `extra_configs`, \
                    otherwise, a `ValueError` will be raised.
        Returns:
            InferenceConfig: an InferenceConfig object initialized from the config file
        """
        raise NotImplementedError("TODO: Assignment5 - Task2")
    
    