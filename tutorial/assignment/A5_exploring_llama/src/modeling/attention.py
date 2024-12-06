from typing import Optional, Tuple
from enum import Enum

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# from assignment1 implementations
from .norm import GroupRMSNorm


class AttnQKVPackFormat(Enum):
    QKV = "qkv_packed"
    Q_KV = "q_kv_packed"
    Q_K_V = "q_k_v_packed"


class AttnQKVLayout(Enum):
    BSHD = "bshd"
    SBHD = "sbhd"
    THD = "thd"


class OfflineSlidingWindowAttn(nn.Module):
    """Offline Sliding-Window Attention module
    This is a generalized variant of standard self-attention equipped with the sliding-window trick \
        to make use of spatial locality in language for computational efficiency, \
        with applying other methods to improve stability.
    """
    def __init__(
        self,
        head_dim: int,
        num_q_head: int,
        num_kv_head: int,
        qkv_pack_format: AttnQKVPackFormat = AttnQKVPackFormat.Q_K_V,
        qkv_layout: AttnQKVLayout = AttnQKVLayout.BSHD,
        window_size: Optional[int] = None,
        causal: bool = False,
        softmax_dropout_rate: float = 0.0,
        softmax_dropout_seed: int = 42,
        softmax_scale: Optional[float] = None,
        softmax_cap: Optional[float] = None,
        softmax_temp: float = 1.0,
        softmax_clip_range: Tuple[float, float] = (0., 1.),
        apply_qk_norm: bool = False,
        group_size: Optional[int] = None,
        eps: float = 1e-5,
        init_range: tuple = (-1.0, 1.0),
        init_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Offline Sliding-Window Attention module
        
        Args:
            head_dim(int): head dimension size
            num_q_head(int): number of query heads
            num_kv_head(int): number of key/value heads
            qkv_pack_format(AttnQKVPackFormat, default = "q_k_v_packed"): qkv packed format
            qkv_layout(AttnQKVLayout, default = "bshd"): qkv shape layout
            window_size(int, default = None): window size
            causal(bool, default = False): if True, then apply causal masking as a prior to only allow unidirectional self-attention, otherwise bidirectional
            softmax_dropout_rate(float, default = 0.0): dropout probability for the softmax probs
            softmax_dropout_seed(int, default = 42): random seed for softmax drooput
            softmax_scale(float, default = None): softmax scale factor, if None, then applying the standard value: 1/√d
            softmax_cap(float, default = None): softmax capping to control the magnitude of the logits, if None, then NO capping is applied
            softmax_temp(float, default = 1.0): softmax temperature to control the sharpness of the distribution, only apply when softmax_cap is None
            softmax_clip_range(float, default = (0.0, 1.0): the range for softmax clipping to prevent the outliers from growing further
            apply_qk_norm(bool, default = False): if True, then apply qk norm
            group_size(int, optional, default = None): group size to split hidden size of query / key for GroupRMSNorm, if None, then set it to `head_dim`, if applying qk norm
            eps(float, default = 1e-5): epsilon for GroupRMSNorm, if applying qk norm
            init_range(tuple, default = (-1.0, 1.0)): the range of the initialization uniform distribution for GroupRMSNorm, if applying qk norm
            init_seed(int, default = 42): initialization seed for GroupRMSNorm, if applying qk norm
            dtype(torch.dtype, default = torch.float32): parameter dtype for GroupRMSNorm, if applying qk norm
            device(str, default = "cpu"): parameter device for GroupRMSNorm, if applying qk norm
        """
        super().__init__()
        raise NotImplementedError("TODO: Assignment3 - Task1")
    
    def forward(
        self,
        q: torch.Tensor,
        k: Optional[torch.Tensor] = None,
        v: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """The forward pass of Offline Sliding-Window Attention module
        
        Args:
            q(torch.Tensor): query tensor, or query-key-value packed tensor if the qkv_pack_format is "qkv_packed"
            k(Optional[torch.Tensor], default = None): key tensor, or key-value packed tensor if the qkv_pack_format is "q_kv_packed", or None if qkv_pack_format is "qkv_packed"
            v(Optional[torch.Tensor], default = None): value tensor if the qkv_pack_format is "q_k_v_packed", otherwise None
            cu_seqlens_q(Optional[torch.Tensor], default = None): cumulative sequence lengths for query tensor, with shape: [batch_size + 1, ]
            cu_seqlens_k(Optional[torch.Tensor], default = None): cumulative sequence lengths for key tensor, with shape: [batch_size + 1, ]
        Returns:
            torch.Tensor: output tensor o, with the same shape as q
        """
        raise NotImplementedError("TODO: Assignment3 - Task1")
    
    def reset_parameters(self):
        """Initialize the optional q, k norm parameters of Offline Sliding-Window Attention module"""
        raise NotImplementedError("TODO: Assignment3 - Task1")
    

class OnlineSlidingWindowAttn(OfflineSlidingWindowAttn):
    """Online Sliding-Window Attention module
    This is a online version of Offline Sliding-Window Attention module \
        which only apply attention on a block of q, k, v in "bshd" layout and "q_k_v_packed" format \
            and update the global o with the local block of o using lse
    """
    def __init__(
        self,
        seqlen_q: int,
        seqlen_kv: int,
        block_size_q: int,
        block_size_kv: int,
        head_dim: int,
        num_q_head: int,
        num_kv_head: int,
        window_size: Optional[int] = None,
        causal: bool = False,
        softmax_scale: Optional[float] = None,
        softmax_cap: Optional[float] = None,
        softmax_temp: float = 1.0,
        apply_qk_norm: bool = False,
        group_size: Optional[int] = None,
        eps: float = 1e-5,
        init_range: tuple = (-1.0, 1.0),
        init_seed: int = 42,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
    ):
        """Initialize Online Sliding-Window Attention module
        
        Args:
            seqlen_q(int): the sequence length of q
            seqlen_kv(int): the sequence length of kv
            block_size_q(int): the block size of q
            block_size_kv(int): the block size of kv
            head_dim(int): head dimension size
            num_q_head(int): number of query heads
            num_kv_head(int): number of key/value heads
            window_size(int, default = None): window size
            causal(bool, default = False): if True, then apply causal masking as a prior to only allow unidirectional self-attention, otherwise bidirectional
            softmax_scale(float, default = None): softmax scale factor, if None, then applying the standard value: 1/√d
            softmax_cap(float, default = None): softmax capping to control the magnitude of the logits, if None, then NO capping is applied
            softmax_temp(float, default = 1.0): softmax temperature to control the sharpness of the distribution, only apply when softmax_cap is None
            apply_qk_norm(bool, default = False): if True, then apply qk norm
            group_size(int, optional, default = None): group size to split hidden size of query / key for GroupRMSNorm, if None, then set it to `head_dim`, if applying qk norm
            eps(float, default = 1e-5): epsilon for GroupRMSNorm, if applying qk norm
            init_range(tuple, default = (-1.0, 1.0)): the range of the initialization uniform distribution for GroupRMSNorm, if applying qk norm
            init_seed(int, default = 42): initialization seed for GroupRMSNorm, if applying qk norm
            dtype(torch.dtype, default = torch.float32): parameter dtype for GroupRMSNorm, if applying qk norm
            device(str, default = "cpu"): parameter device for GroupRMSNorm, if applying qk norm
        """
        super().__init__(
            head_dim=head_dim,
            num_q_head=num_q_head,
            num_kv_head=num_kv_head,
            window_size=window_size,
            causal=causal,
            softmax_scale=softmax_scale,
            softmax_cap=softmax_cap,
            softmax_temp=softmax_temp,
            apply_qk_norm=apply_qk_norm,
            group_size=group_size,
            eps=eps,
            init_range=init_range,
            init_seed=init_seed,
            dtype=dtype,
            device=device,
        )
        raise NotImplementedError("TODO: Assignment3 - Task2")
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        global_o: torch.Tensor,
        global_lse: torch.Tensor,
        block_idx_q: int,
        block_idx_kv: int,
    ) -> None:
        """The forward pass of Offline Sliding-Window Attention module
        
        Args:
            q(torch.Tensor): query tensor, with shape: [batch_size, block_size_q, num_q_head, head_dim]
            k(torch.Tensor): key tensor, with shape: [batch_size, block_size_kv, num_kv_head, head_dim]
            v(torch.Tensor): value tensor, with shape: [batch_size, block_size_kv, num_kv_head, head_dim]
            global_o(torch.Tensor): global output tensor to be updated inplace, with shape: [batch_size, seqlen_q, num_q_head, head_dim]
            global_lse(torch.Tensor): global lse tensor to be updated inplace, with shape: [batch_size, num_q_head, seqlen_q]
            block_idx_q(int): the block index of q
            block_idx_kv(int): the block index of kv
        """
        raise NotImplementedError("TODO: Assignment3 - Task2")
    