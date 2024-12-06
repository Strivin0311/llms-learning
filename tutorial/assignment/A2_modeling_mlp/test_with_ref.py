import sys
sys.path.insert(0, ".")

import pytest

import torch
from torch.testing import assert_close

from ref.modeling import (
    DenseMLPWithLoRA as DenseMLPWithLoRARef,
    SparseMLPWithLoRA as SparseMLPWithLoRARef,
    MLPActivationType as MLPActivationTypeRef,
)

from src.modeling import (
    DenseMLPWithLoRA,
    SparseMLPWithLoRA,
    MLPActivationType,
)


# constants for all toy test cases
ATOL = 1e-3
RTOL = 1e-3
SEED = 42
PARAM_DEVICE = "cpu"
PARAM_DTYPE = torch.float32

# mapping from ref mlp_activation_type to student mlp_activation_type
mlp_activation_type_ref_to_student = {
    MLPActivationTypeRef.SIGMOID: MLPActivationType.SIGMOID,
    MLPActivationTypeRef.SILU: MLPActivationType.SILU,
    MLPActivationTypeRef.GELU: MLPActivationType.GELU,
    MLPActivationTypeRef.BILINEAR: MLPActivationType.BILINEAR,
    MLPActivationTypeRef.RELU: MLPActivationType.RELU,
}

# configs for each toy test case
toy_test_cases = {
    "task1": {
        "case1": {
            "b": 1,
            "s": 2,
            "h": 4,
            "ffh": 8,
            
            "activation_type": MLPActivationTypeRef.SILU,
            
            "r": 2,
            "alpha": None,
            "dropout": 0.0,
            
            "activation_dtype": torch.bfloat16,
            "activation_device": "cpu",
        }
    },
    "task2": {
        "case1": {
            "b": 2,
            "s": 3,
            "h": 4,
            "ffh": 8,
            
            "activation_type": MLPActivationTypeRef.SILU,
            
            "ne": 4,
            "k": 2,
            
            "rank": 1,
            "world_size": 2,
            
            "init_mean": 0.1,
            "init_std": 1.1,
            
            "r": 2,
            "alpha": None,
            "dropout": 0.0,
            
            "activation_dtype": torch.bfloat16,
            "activation_device": "cpu",
        }
    },
}


@pytest.mark.parametrize(
    "case_key, case_config",
    toy_test_cases["task1"].items(),
)
def test_task1(case_key, case_config):
    # set hyper parameters
    b, s, h, ffh = case_config["b"], case_config["s"], case_config["h"], case_config["ffh"]
    activation_type = case_config["activation_type"]
    r, alpha, dropout = case_config["r"], case_config.pop("alpha", None), case_config.pop("dropout", 0.0)
    init_base_seed, lora_init_base_seed, lora_dropout_seed = case_config.pop("init_base_seed", SEED), \
        case_config.pop("lora_init_base_seed", SEED), \
        case_config.pop("lora_dropout_seed", SEED)
    atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)
    activation_dtype, param_dtype = case_config["activation_dtype"], case_config.pop("param_dtype", PARAM_DTYPE)
    activation_device, param_device = case_config["activation_device"], case_config.pop("param_device", PARAM_DEVICE)

    # construct the input tensor
    torch.manual_seed(init_base_seed)
    input = torch.randn(b, s, h, dtype=activation_dtype, device=activation_device)
    
    # instantiate the reference module
    dense_mlp_ref = DenseMLPWithLoRARef(
        hidden_size=h,
        ffh_size=ffh,
        activation_type=activation_type,
        init_base_seed=init_base_seed,
        lora_rank=r,
        lora_alpha=alpha,
        lora_dropout_rate=dropout,
        lora_dropout_seed=lora_dropout_seed,
        lora_init_base_seed=lora_init_base_seed,
        dtype=param_dtype,
        device=param_device,
    )
    
    # apply the forward pass to get the reference output tensor
    output_ref = dense_mlp_ref(input.clone())
    
    # instantiate the student's module
    dense_mlp = DenseMLPWithLoRA(
        hidden_size=h,
        ffh_size=ffh,
        activation_type=mlp_activation_type_ref_to_student[activation_type],
        init_base_seed=init_base_seed,
        lora_rank=r,
        lora_alpha=alpha,
        lora_dropout_rate=dropout,
        lora_dropout_seed=lora_dropout_seed,
        lora_init_base_seed=lora_init_base_seed,
        dtype=param_dtype,
        device=param_device,
    )
    
    # apply the forward pass to get student's output tensor
    output = dense_mlp(input)
    
    # check if the output tensor is correct
    assert_close(output, output_ref, atol=atol, rtol=rtol)
    

@pytest.mark.parametrize(
    "case_key, case_config",
    toy_test_cases["task2"].items(),
)
def test_task2(case_key, case_config):
    # set hyper parameters
    b, s, h, ffh = case_config["b"], case_config["s"], case_config["h"], case_config["ffh"]
    activation_type = case_config["activation_type"]
    ne, k, rank, world_size = case_config["ne"], case_config["k"], case_config["rank"], case_config["world_size"]
    init_mean, init_std = case_config["init_mean"], case_config["init_std"]
    r, alpha, dropout = case_config["r"], case_config.pop("alpha", None), case_config.pop("dropout", 0.0)
    init_base_seed, lora_init_base_seed, lora_dropout_seed = case_config.pop("init_base_seed", SEED), \
        case_config.pop("lora_init_base_seed", SEED), \
        case_config.pop("lora_dropout_seed", SEED)
    atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)
    activation_dtype, param_dtype = case_config["activation_dtype"], case_config.pop("param_dtype", PARAM_DTYPE)
    activation_device, param_device = case_config["activation_device"], case_config.pop("param_device", PARAM_DEVICE)

    # construct the input tensor
    torch.manual_seed(init_base_seed + 1)
    input = torch.randn(b, s, h, dtype=activation_dtype, device=activation_device)
    
    # instantiate the reference module
    sparse_mlp_ref = SparseMLPWithLoRARef(
        hidden_size=h,
        ffh_size=ffh,
        activation_type=activation_type,
        num_experts=ne,
        moe_topk=k,
        rank=rank,
        world_size=world_size,
        init_mean=init_mean,
        init_std=init_std,
        init_base_seed=init_base_seed,
        lora_rank=r,
        lora_alpha=alpha,
        lora_dropout_rate=dropout,
        lora_dropout_seed=lora_dropout_seed,
        lora_init_base_seed=lora_init_base_seed,
        dtype=param_dtype,
        device=param_device,
    )
    
    # apply the forward pass to get the reference output tensor
    output_ref = sparse_mlp_ref(input.clone())
    
    # instantiate the student's module
    sparse_mlp = SparseMLPWithLoRA(
        hidden_size=h,
        ffh_size=ffh,
        activation_type=mlp_activation_type_ref_to_student[activation_type],
        num_experts=ne,
        moe_topk=k,
        rank=rank,
        world_size=world_size,
        init_mean=init_mean,
        init_std=init_std,
        init_base_seed=init_base_seed,
        lora_rank=r,
        lora_alpha=alpha,
        lora_dropout_rate=dropout,
        lora_dropout_seed=lora_dropout_seed,
        lora_init_base_seed=lora_init_base_seed,
        dtype=param_dtype,
        device=param_device,
    )
    
    # apply the forward pass to get student's output tensor
    output = sparse_mlp(input)
    
    # check if the output tensor is correct
    assert_close(output, output_ref, atol=atol, rtol=rtol)


if __name__ == "__main__":
    pytest.main()
    