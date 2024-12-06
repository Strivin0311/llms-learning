from typing import Dict, Any, List, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..config import (
    config_dataclass,
    make_required_field,
    make_fixed_field,
)

from ..models.base import BaseTokenizer

from .base import BaseDatasetConfig

from .qa import QADataset


@config_dataclass
class ChatDatasetConfig(BaseDatasetConfig):
    """Dataset Configurations Dataclass for Chatbot Tasks"""
    
    conversations_key: str = make_fixed_field("conversations")
    role_key: str = make_fixed_field("role")
    content_key: str = make_fixed_field("content")
    
    user_role_value: str = make_fixed_field("user")
    bot_role_value: str = make_fixed_field("chatbot")
    
    user_role_prefix: str = make_fixed_field("USER")
    bot_role_prefix: str = make_fixed_field("CHATBOT")


class ChatDataset(QADataset):
    """Dataset Class for Chatbot Tasks"""
    
    def __init__(
        self,
        config: ChatDatasetConfig,
        tokenizer: BaseTokenizer,
        data_files: Union[str, List[str]],
    ):
        """Initialize ChatDataset module
        Args:
            config (ChatDatasetConfig): chat dataset configuration dataclass object
            tokenizer (BaseTokenizer): tokenizer module
            data_files (Union[str, List[str]]): path to the file(s) with the data in .jsonl format
        """
        raise NotImplementedError("TODO: Assignment5 - Task3")
    