from typing import Tuple, Optional

import torch
import torch.nn.functional as F


def matmul_with_importance(
    input: torch.Tensor,
    weight: torch.Tensor,
    probs: torch.Tensor,
    grad_output: Optional[torch.Tensor] = None,
    num_heads: int = 1,
    top_p: float = 1.0,
    top_k: Optional[int] = None,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    """matmul input and weight and return output (with optional grad_input, grad_weight whenever grad_output is given) 
    where only the important elements of the input tensor can be computed and gathered to the output tensor
    decided by the importance probability tensor, tuned by top_p and top_k
    
    Args:
        input (torch.Tensor): input tensor in the range of [-1, 1], with shape: [batch_size, seq_len, hidden_size]
        weight (torch.Tensor): weight tensor in the range of [-1, 1], with shape: [hidden_size, embed_size]
        probs (torch.Tensor): probability tensor in the range of [0, 1], with shape: [batch_size, seq_len]
        grad_output (Optional[torch.Tensor], optional): gradient for the output tensor, with shape: [t, hidden_size]. Defaults to None.
        num_heads (int): number of heads to split hidden_size
        top_p (float, [0., 1.]): only the elements with the probability equal or higher than top_p are important ones
        top_k (int, [1, ..., seq_len], optional): only the elements with the top_k highest probability are important ones
    
    Returns:
        output (torch.Tensor): output tensor, with shape: [t, num_heads, embed_size]
        grad_input (torch.Tensor, optional): gradient for the input tensor if grad_output is given, otherwise None
        grad_weight (torch.Tensor, optional): gradient for the weight tensor if grad_output is given, otherwise None
    """
    raise NotImplementedError("TODO: Assignment0 - Task1")