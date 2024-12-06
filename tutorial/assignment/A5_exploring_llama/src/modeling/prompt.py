
from typing import Dict, Optional
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import find_format_keys


class BatchLayout(Enum):
    """Batch Layout Enum"""
    CONCAT = "concat"
    STACK = "stack"


class PaddingSide(Enum):
    """Padding Side Enum"""
    LEFT = "left"
    RIGHT = "right"


class TruncateSide(Enum):
    """Truncate Side Enum"""
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


class PromptType(Enum):
    """Prompt Types Enum"""
    
    SYSTEM = "system"
    CONTEXT = "context"
    QUERY = "query"
    RESPONSE = "response"
    PROMPT = "prompt" # NOTE: prompt = system + context + query
    ALL = "all" # NOTE: all = prompt + response
    

class PromptTemplate(nn.Module):
    """Prompt Template module"""
    
    def __init__(self, template_str: str = ""):
        """Initialize Prompt Template module
        
        Args:
            template_str (str): the template string with the format: "....{key1}...{key2}..."
        """
        super().__init__()
        raise NotImplementedError("TODO: Assignment5 - Task2")
    
    def keys(self) -> Dict[str, Optional[str]]:
        """Get the keys with its default values of the prompt template as a dictionary
        NOTE: if any key has not been set with default value, then use `None` as a placeholder
        """
        raise NotImplementedError("TODO: Assignment5 - Task2")
    
    def set_default(self, **kwargs: Optional[Dict[str, str]]) -> None:
        """Set the default values of the prompt template keys"""
        raise NotImplementedError("TODO: Assignment5 - Task2")
    
    def forward(self, **kwargs: Optional[Dict[str, str]]) -> str:
        """Set the prompt template keys with the given keyword argument to get the formatted prompt
        NOTE:
            1. if certain prompt template key has not been set with its default value, then its corresponding kwarg should be provided
            2. if certain key in the kwargs is not found in the keys of the prompt template, just ignore it
        """
        raise NotImplementedError("TODO: Assignment5 - Task2")

