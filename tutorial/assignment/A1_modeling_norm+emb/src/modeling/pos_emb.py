import torch
import torch.nn as nn
import torch.nn.functional as F

from ..functional import apply_rotary_pos_emb


class NTKAwareRoPE(nn.Module):
    """NTK-aware RoPE module
    This is a series variants of the RoPE modules based on NTK theory to enhance its extrapolation ability.
    """
    
    def __init__(
        self, 
        dim: int, 
        max_seq_len: int,
        base: int = 10000,
        ratio: int = 1,
        dynamic: bool = False,
        dtype: torch.dtype = torch.float32,
        device: str = 'cpu',
    ) -> None:
        """Initialize NTK-aware RoPE Module
        
        Args:
            dim (int): The dimension of the RoPE
            max_seq_len (int): The maximum sequence length used in training
            base (int, optional): The base of the NTK. Defaults to 10000.
            ratio (int, optional): The ratio of the NTK. Defaults to 1.
            dynamic (bool, optional): Whether to use dynamic mode. Defaults to False.
            dtype (torch.dtype, optional): The dtype of the RoPE. Defaults to torch.float32.
            device (str, optional): The device of the RoPE. Defaults to 'cpu'.
        """
        super().__init__()
        raise NotImplementedError("TODO: Assignment1 - Task3")
        
    def forward(self, input: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """The forward pass of the NTK-aware RoPE module
        
        Args:
            input(torch.Tensor): input tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
            offset(int, optional): The offset of the starting position index of the input tensor. Defaults to 0.
        
        Returns:
            output(torch.Tensor): embedded output tensor, with shape: [batch_size, seq_len, num_heads, head_dim]
        """
        raise NotImplementedError("TODO: Assignment1 - Task3")