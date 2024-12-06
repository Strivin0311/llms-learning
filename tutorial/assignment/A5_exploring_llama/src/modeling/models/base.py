
from abc import ABC, abstractmethod
from typing import Optional, List, Union, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..transformer import TransformerDecoderKVCache


class BaseTokenizer(nn.Module, ABC):
    """Base Tokenizer module
    This is the basic abstract tokenizer module for any instantiable tokenizer module (specified for some language models) to inherit, \
        which defines some abstract methods required to be implemented as the common APIs
    """
    
    @abstractmethod
    def encode(
        self,
        prompt: Union[str, List[str]],
    ) -> List[torch.LongTensor]:
        """Encode a single or a list of prompt(s) into a list of tensors of encoded ids
        
        Args:
            prompt (Union[str, List[str]]): a single or a list of prompt(s) to be encoded
        
        Returns:
            List[torch.LongTensor]: a list of 1-dim tensors of encoded ids, each with shape: [seq_len,], \
                NOTE: if the input is a single prompt, the output will be a list with length 1
        """
    
    @abstractmethod
    def decode(
        self,
        encoded_ids: Union[torch.LongTensor, List[torch.LongTensor]],
    ) -> List[str]:
        """Decode a single or a list of tensor(s) of encoded ids into a list of prompts
        
        Args:
            encoded_ids (Union[torch.LongTensor, List[torch.LongTensor]]): a single or a list of tensor(s) of encoded ids to be decoded, \
                NOTE: each tensor is a 1-dim tensor with shape: [seq_len,]
        
        Returns:
            List[str]: a list of prompt strings, each with length: `seq_len` (stripped the special tokens), \
                NOTE: if the input is a single tensor of encoded ids, the output will be a list with length 1
        """

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """The vocab size of the tokenizer"""
    
    @property
    @abstractmethod
    def pad_token(self) -> str:
        """The pad token of the tokenizer"""
    
    @property
    @abstractmethod
    def pad_id(self) -> int:
        """The pad token id of the tokenizer"""
    
    @property
    @abstractmethod
    def bos_token(self) -> str:
        """The bos token of the tokenizer"""
        
    @property
    @abstractmethod
    def bos_id(self) -> int:
        """The bos token id of the tokenizer"""
    
    @property
    @abstractmethod
    def eos_token(self) -> str:
        """The eos token of the tokenizer"""    
    
    @property
    @abstractmethod
    def eos_id(self) -> int:
        """The eos token id of the tokenizer"""
    

class BaseModel(nn.Module, ABC):
    """Base Model module
    This is the basic abstract model module for any instantiable language model module to inherit, \
        which defines some abstract methods required to be implemented as the common APIs
    """
    
    @abstractmethod
    def forward(
        self,
        input_ids: torch.LongTensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        labels: Optional[torch.LongTensor] = None,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """The forward pass of the Base Model module
        
        Args:
            input_ids(torch.LongTensor): the vocab ids for the input, either as a training sample or as a query prompt, with shape: [batch_size, seq_len]
            cu_seqlens(torch.Tensor, optional): cumulative sequence lengths for input tensor, with shape: [inner_batch_size + 1, ]
            labels(Optional[torch.LongTensor]): the vocab ids for the training labels, with the same shape as `input_ids` (and the `ignore_index` is set to `-100` as default by pytorch)
            NOTE: if `cu_seqlens` is not None, then the `batch_size` in the shape of `input_ids` is ensured to be `1` to remain the 2-dim shape, \
                while the real `batch_size` is inferred from `cu_seqlens` (i.e. `inner_batch_size`) since the inner sequences are concatenated along the `seqlen` dim.
            temperature (float, optional): temperature for softmax during inference, default to 1.0, in the range of (0., +inf)
        Returns:
            torch.Tensor: output tensor, whose value depends on the model's state:
                1. for training, the value is just the cross-entroy scalar loss by `mean` reduction between the vocab logits and the given labels
                2. for inference, the value is the next-token probability distribution for every sequence in the batch, with shape: [inferred_batch_size, vocab_size]
            NOTE: the `inferred_batch_size` is inferred from `cu_seqlens` if provided (i.e. `inner_batch_size`), otherwise from `input_ids` (i.e. `batch_size`)
        """

    @abstractmethod
    def get_kv_cache(self) -> TransformerDecoderKVCache:
        """Get the TransformerDecoderKVCache object managing the kv cache memory"""

    @abstractmethod
    def set_kv_cache(self, kv_cache: TransformerDecoderKVCache):
        """Set the TransformerDecoderKVCache object managing the kv cache memory"""
        
    @abstractmethod
    def reset_kv_cache(self):
        """Reset the TransformerDecoderKVCache object managing the kv cache memory"""
        
    @abstractmethod
    def num_parameters(
        self,
        learnable_only: bool = False, 
        unit: Literal["1", "K", "M", "B"] = "1"
    ) -> float:
        """Compute the number of (learnable) parameters in the Base Model module
        
        Args:
            learnable_only(bool, optional): whether to count only learnable parameters or not, default to False
            unit(str, optional): unit of the number of parameters, default to '1' for "1", \
                other options include 'K' for "1 thousand", 'M' for "1 million", 'B' for "1 billion"
        Returns:
            float: the number of (learnable) parameters in the Base Model module in the specified unit
        """
        
    @abstractmethod
    def num_memory_footprint(
        self,
        unit: Literal["B", "KB", "MB", "GB"] = "B"
    ) -> float:
        """Compute the theoretical memory footprint of the Base Model module's parameters
        
        Args:
            unit(str, optional): unit of the memory footprint, default to 'B' for "1 byte", \
                other options include 'KB' for "1 kilobyte", 'MB' for "1 megabyte", 'GB' for "1 gigabyte"
                
        Returns:
            float: the theoretical memory footprint of the Base Model module's parameters in the specified unit
        """
        
    

