from typing import Optional, Tuple, Sequence, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import ProcessGroup

# from assignment1 implementations
from .vocab_emb import ParallelVocabEmbedding
from .pos_emb import NTKAwareRoPE
from .norm import GroupRMSNorm

# from assignment2 implementations
from .mlp import (
    MLPActivationType,
    DenseMLPWithLoRA,
    SparseMLPWithLoRA,
)

# from assignment3 implementations
from .attention import (
    AttnQKVPackFormat,
    AttnQKVLayout,
    OfflineSlidingWindowAttn,
    OnlineSlidingWindowAttn,
)

from .config import (
    BaseConfig,
    config_dataclass,
    make_required_field,
    make_fixed_field,
)


@config_dataclass
class TransformerConfig(BaseConfig):
    """Transformer Configurations Dataclass"""
    
    # common transformer configurations
    num_layers: int = make_required_field()
    hidden_size: int = make_required_field()
    ffh_size: int = make_required_field()
    max_seq_len: int = make_required_field()
    param_dtype: torch.dtype = torch.float32
    param_device: str = "cpu"
    init_base_seed: int = 42
    
    # fixed distributed configurations
    rank: int = make_fixed_field(0)
    world_size: int = make_fixed_field(1)
    process_group: Optional[ProcessGroup] = make_fixed_field(None)
    
    # vocab embedding configurations
    vocab_size: int = make_required_field()
    vocab_init_mean: float = 0.0
    vocab_init_std: float = 1.0
    
    # positional embedding configurations
    rope_base: int = 10000
    rope_ratio: int = 1
    rope_dynamic: bool = False
    
    # normalization configurations
    group_size: Optional[int] = None
    eps: float = 1e-5
    norm_init_range: tuple = (-1.0, 1.0)
    
    # projection configurations
    proj_init_seed: int = 42
    proj_init_mean: float = 0.0
    proj_init_std: float = 1.0
    lm_head_tied: bool = False
    
    # attention configurations
    online_attn_block_size: Optional[int] = None # NOTE: if None, then use offline mode, otherwise use online mode
    head_dim: int = make_required_field()
    num_q_head: int = make_required_field()
    num_kv_head: int = make_required_field()
    qkv_pack_format: AttnQKVPackFormat = AttnQKVPackFormat.Q_K_V
    qkv_layout: AttnQKVLayout = AttnQKVLayout.BSHD
    window_size: Optional[int] = None
    causal: bool = False
    softmax_dropout_rate: float = 0.0
    softmax_dropout_seed: int = 42
    softmax_scale: Optional[float] = None
    softmax_cap: Optional[float] = None
    softmax_temp: float = 1.0
    softmax_clip_range: Tuple[float, float] = (0., 1.)
    apply_qk_norm: bool = False
    qk_norm_group_size: Optional[int] = None # NOTE: the other configurations of qk norm are the same as the ones of normalization above
    
    # dense mlp configurations
    activation_type: MLPActivationType = MLPActivationType.SILU
    lora_rank: int = 0
    lora_alpha: Optional[float] = None
    lora_dropout_rate: float = 0.0
    lora_dropout_seed: int = 42
    lora_init_base_seed: int = 42
    
    # sparse mlp configurations (optional)
    num_experts: Optional[int] = None # NOTE: if None, then use dense mlp, otherwise use sparse mlp
    moe_topk: int = 1
    gate_init_mean: float = 0.0
    gate_init_std: float = 1.0


class TransformerDecoderKVCache(nn.Module):
    """Transformer KV cache module
    This is a simple module to manage cached past key-value pairs for each transformer decoder layer \
        tradeoff memory footprint for avoiding redundant computation during inference.
    """
    def __init__(
        self,
        qkv_layout: AttnQKVLayout = AttnQKVLayout.BSHD,
        num_layers: int = 1,
    ):
        """Initialize Transformer KV cache module
        
        Args:
            qkv_layout (AttnQKVLayout, optional): Layout of the q, k, v tensors. Defaults to AttnQKVLayout.BSHD.
            num_layers (int, optional): Number of transformer layers. Defaults to 1.
        """
        super().__init__()
        raise NotImplementedError("TODO: Assignment4 - Task1")

    def has(self, layer_idx: int) -> bool:
        """Check if cached past key-value pairs exist for a specific layer
        
        Args:
            layer_idx (int): Layer index

        Returns:
            bool: True if cached past key-value pairs exist for the layer, False otherwise
        """
        raise NotImplementedError("TODO: Assignment4 - Task1")

    def get(
        self, 
        layer_idx: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """Get cached past key-value pairs with their optional cumulative sequence lengths for a specific layer
        
        Args:
            layer_idx (int): Layer index

        Returns:
            Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]: (k, v, optional cu_seqlens)
            
        Raises:
            KeyError: If cached past key-value pairs do not exist for the layer
        """
        raise NotImplementedError("TODO: Assignment4 - Task1")

    def set(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        """Set cached past key-value pairs with their optional cumulative sequence lengths for a specific layer
        
        Args:
            layer_idx (int): Layer index
            k (torch.Tensor): Key tensor to set
            v (torch.Tensor): Value tensor to set
            cu_seqlens (Optional[torch.Tensor], optional): Cumulative sequence lengths for the key-value pairs to set. Defaults to None.
            NOTE: The `cu_seqlens` must be provided if the `qkv_layout` is AttnQKVLayout.THD
        """
        raise NotImplementedError("TODO: Assignment4 - Task1")

    def append(
        self,
        layer_idx: int,
        k: torch.Tensor,
        v: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> None:
        """Dynamically append current cached past key-value pairs with their optional cumulative sequence lengths to the existing ones for a specific layer
        
        Args:
            layer_idx (int): Layer index
            k (torch.Tensor): Key tensor to append
            v (torch.Tensor): Value tensor to append
            cu_seqlens (Optional[torch.Tensor], optional): Cumulative sequence lengths for the key-value pairs to append. Defaults to None.
            NOTE: The `cu_seqlens` must be provided if the `qkv_layout` is AttnQKVLayout.THD, \
                and all of the pass-in arguments should be consistent with the existing ones.
        """
        raise NotImplementedError("TODO: Assignment4 - Task1")
    
    def reset(self):
        """Clear the cache memory and reset to the initial state
        """
        raise NotImplementedError("TODO: Assignment4 - Task1")
    

class TransformerDecoderLayer(nn.Module):
    """Transformer Decoder Layer module
    This is a variant of transformer decoder layer, consisting of two sub-layers: \
            one offline / online self-attention layer, along with qkv projection, ntk-aware rope and out projection, \
            and one dense / sparse feed-forward mlp layer, supporting LoRA adaption intrinsically, \
        which are concatenated sequentially with residual connections and group rms normalization.
    """
    
    def __init__(
        self,
        config: TransformerConfig,
        layer_idx: int = 0,
    ):
        """Initialize Transformer Decoder Layer module
        
        Args:
            config (TransformerConfig): transformer configuration
            layer_idx (int): layer index, in the range of [0, num_layers). Defaults to 0.
        """
        super().__init__()
        raise NotImplementedError("TODO: Assignment4 - Task2")
    
    def forward(
        self,
        input: torch.Tensor,
        cu_seqlens: Optional[torch.Tensor] = None,
        kv_cache: Optional[TransformerDecoderKVCache] = None,
    ) -> torch.Tensor:
        """The forward pass of Transformer Decoder Layer module
        
        Args:
            input(torch.Tensor): input hidden states tensor, with shape: [batch_size, seq_len, hidden_size]
            cu_seqlens(torch.Tensor, optional): cumulative sequence lengths for input tensor, with shape: [inner_batch_size + 1, ]
            kv_cache(Optional[TransformerDecoderKVCache], default = None): transformer kv cache, to retrieve / update past key and value during inference, \
                if None, then no kv cache (i.e. during training)
            NOTE: if `cu_seqlens` is not None, then the `batch_size` in the shape of `input` is ensured to be `1` to remain the 3-dim shape, \
                while the real `batch_size` is inferred from `cu_seqlens` (i.e. `inner_batch_size`) since the inner sequences are concatenated along the `seqlen` dim.
        Returns:
            torch.Tensor: output hidden states tensor, with the same shape as input
        """
        raise NotImplementedError("TODO: Assignment4 - Task2")
    
    def reset_parameters(self):
        """Initialize learnable parameters for Transformer Decoder Layer module"""
        raise NotImplementedError("TODO: Assignment4 - Task2")


class TransformerDecoderBlock(nn.Module):
    """Transformer Decoder Block module
    
    This is a standard decoder-only transformer block for language modeling, \
        which mainly consists of a sequence of transformer decoder layers, \
        transforming the hidden states of input token ids initialized from vocab embedding, \
        and finally returning the vocab logits with a lm head projection.
    """
    
    def __init__(
        self,
        config: TransformerConfig,
    ):
        """Initialize Transformer Decoder Block module
        
        Args:
            config (TransformerConfig): transformer configuration
        """
        super().__init__()
        raise NotImplementedError("TODO: Assignment4 - Task3")
    
    def forward(
        self,
        input_ids: torch.LongTensor,
        cu_seqlens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The forward pass of Transformer Decoder Block module
        
        Args:
            input_ids(torch.LongTensor): the vocab ids for the input, with shape: [batch_size, seq_len]
            cu_seqlens(torch.Tensor, optional): cumulative sequence lengths, with shape: [inner_batch_size + 1, ]
            NOTE: if `cu_seqlens` is not None, then the `batch_size` in the shape of `input_ids` is ensured to be `1` to remain the 2-dim shape, \
                while the real `batch_size` is inferred from `cu_seqlens` (i.e. `inner_batch_size`) since the inner sequences are concatenated along the `seqlen` dim.
        Returns:
            torch.Tensor: output tensor as vocab logits, with shape: [batch_size, seq_len, vocab_size]
        """
        raise NotImplementedError("TODO: Assignment4 - Task3")
    
    def get_kv_cache(self) -> TransformerDecoderKVCache:
        """Get the TransformerDecoderKVCache object managing the kv cache memory"""
        raise NotImplementedError("TODO: Assignment4 - Task3")
    
    def set_kv_cache(self, kv_cache: TransformerDecoderKVCache):
        """Set the TransformerDecoderKVCache object managing the kv cache memory"""
        raise NotImplementedError("TODO: Assignment4 - Task3")
    
    def reset_kv_cache(self):
        """Clear the cache memory and reset to the initial state"""
        raise NotImplementedError("TODO: Assignment4 - Task3")
       
    def reset_parameters(self):
        """Initialize learnable parameters for Transformer Decoder Block module"""
        raise NotImplementedError("TODO: Assignment4 - Task3")
     
    def num_parameters(
        self,
        learnable_only: bool = False, 
        unit: Literal["1", "K", "M", "B"] = "1"
    ) -> float:
        """Compute the number of (learnable) parameters in the Llama Model module
        
        Args:
            learnable_only(bool, optional): whether to count only learnable parameters or not, default to False
            unit(str, optional): unit of the number of parameters, default to '1' for "1", \
                other options include 'K' for "1 thousand", 'M' for "1 million", 'B' for "1 billion"
        Returns:
            float: the number of (learnable) parameters in the Llama Model module in the specified unit
        """
        raise NotImplementedError("TODO: Assignment4 - Task3")
    
    def num_memory_footprint(
        self,
        unit: Literal["B", "KB", "MB", "GB"] = "B"
    ) -> float:
        """Compute the theoretical memory footprint of the Llama Model module's parameters
        
        Args:
            unit(str, optional): unit of the memory footprint, default to 'B' for "1 byte", \
                other options include 'KB' for "1 kilobyte", 'MB' for "1 megabyte", 'GB' for "1 gigabyte"
                
        Returns:
            float: the theoretical memory footprint of the Llama Model module's parameters in the specified unit
        """
        raise NotImplementedError("TODO: Assignment4 - Task3")