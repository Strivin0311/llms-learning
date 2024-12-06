import sys
sys.path.insert(0, ".")

import pytest

import torch
from torch.testing import assert_close

from ref import matmul_with_importance as matmul_with_importance_ref

from src import matmul_with_importance

# constants for all toy test cases
ATOL = 1e-3
RTOL = 1e-3
SEED = 42
DEVICE = "cpu"
DTYPE = torch.float32


# configs for each toy test case
toy_test_cases = {
    "case1": {
        "b": 2,
        "s": 5,
        "h": 4,
        "nh": 2,
        "e": 3,
        "top_p": 0.7,
        "top_k": 2,
    }
}


@pytest.mark.parametrize(
    "case_key, case_config",
    toy_test_cases.items(),
)
def test_task1(case_key, case_config):
    # define hyper parameters
    b, s, h, nh, e = case_config["b"], case_config["s"], case_config["h"], case_config["nh"], case_config["e"]
    top_p, top_k = case_config["top_p"], case_config["top_k"]
    seed = case_config.pop("seed", SEED)
    atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)
    device, dtype = case_config.pop("device", DEVICE), case_config.pop("dtype", DTYPE)

    # construct the necessary tensors
    torch.manual_seed(seed)
    input = torch.randn(b, s, h, device=device, dtype=dtype)
    weight = torch.randn(h, e, device=device, dtype=dtype)
    probs = torch.rand(b, s, device=device, dtype=dtype)
    

    #----- test if the function works without grad_output -----#
    output_ref, grad_input_ref, grad_weight_ref = matmul_with_importance_ref(
        input=input.clone(), 
        weight=weight.clone(), 
        probs=probs.clone(),
        num_heads=nh,
        top_p=top_p, 
        top_k=top_k
    )
    output, grad_input, grad_weight = matmul_with_importance(
        input=input, 
        weight=weight, 
        probs=probs,
        num_heads=nh,
        top_p=top_p, 
        top_k=top_k
    )
    
    # check if the output tensor is correct
    assert_close(output, output_ref, atol=atol, rtol=rtol)
    # check if the grad_input tensor is correct
    assert grad_input is None, "grad_input should be None"
    # check if the grad_weight tensor is correct
    assert grad_weight is None, "grad_weight should be None"
    
    #----- test if the function works with grad_output -----#
    torch.manual_seed(seed)
    # grad_output = torch.randn_like(output)
    grad_output = torch.randn(output.size(), dtype=output.dtype, device=output.device)
    output_ref, grad_input_ref, grad_weight_ref = matmul_with_importance_ref(
        input=input.clone(), 
        weight=weight.clone(), 
        probs=probs.clone(),
        grad_output=grad_output.clone(),
        num_heads=nh,
        top_p=top_p, 
        top_k=top_k
    )
    output, grad_input, grad_weight = matmul_with_importance(
        input=input, 
        weight=weight, 
        probs=probs,
        grad_output=grad_output,
        num_heads=nh,
        top_p=top_p, 
        top_k=top_k
    )
    
    # check if the output tensor is correct
    assert_close(output, output_ref, atol=atol, rtol=rtol)
    # check if the grad_input tensor is correct
    assert_close(grad_input, grad_input_ref, atol=atol, rtol=rtol)
    # check if the grad_weight tensor is correct
    assert_close(grad_weight, grad_weight_ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main()
    