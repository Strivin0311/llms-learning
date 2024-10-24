import sys
sys.path.insert(0, ".")

import pytest

import torch
from torch.testing import assert_close

from src.modeling import (
    GroupRMSNorm,
    ParallelVocabEmbedding,
    NTKAwareRoPE,
)


# constants for all toy test cases
ATOL = 1e-3
RTOL = 1e-3
SEED = 42
PARAM_DEVICE = "cpu"
PARAM_DTYPE = torch.float32

# configs for each toy test case
toy_test_cases = {
    "task1": {
        "case1": {
            "b": 1,
            "s": 5,
            "h": 8,
            "gz": 2,
            "eps": 2e-4,
            
            "init_range": (-0.1, 0.1),
            
            "activation_dtype": torch.bfloat16,
            "activation_device": "cpu",
        }
    },
    "task2": {
        "case1": {
            "v": 16,
            "e": 4,
            
            "rank": 1,
            "world_size": 2,
            
            "init_mean": 0.,
            "init_std": 1.,
            
            "activation_device": "cpu",
        }
    },
    "task3": {
        "case1": {
            "b": 1,
            "s": 5,
            "nh": 1,
            "hd": 4,
            
            "ms": 5,
            "base": 10000,
            "k": 1,
            
            "dynamic": False,
            
            "activation_dtype": torch.bfloat16,
            "activation_device": "cpu",
        },
        "case2": {
            "b": 1,
            "s": 9,
            "nh": 1,
            "hd": 4,
            
            "ms": 2,
            "base": 50000,
            "k": 4,
            
            "dynamic": True,
            
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
    b, s, h, gz = case_config["b"], case_config["s"], case_config["h"], case_config["gz"]
    eps, init_range = case_config["eps"], case_config["init_range"]
    init_seed = case_config.pop("init_seed", SEED)
    atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)
    activation_dtype, param_dtype = case_config["activation_dtype"], case_config.pop("param_dtype", PARAM_DTYPE)
    activation_device, param_device = case_config["activation_device"], case_config.pop("param_device", PARAM_DEVICE)
    
    # construct the reference output tensor
    if case_key == "case1":
        output_ref = torch.tensor(
            [[[-0.0505, -0.1040, -0.0305,  0.0505,  0.0238,  0.0182,  0.0630,
                -0.0334],
                [ 0.0986,  0.0476,  0.0312, -0.0417, -0.0244, -0.0176, -0.0684,
                    0.0071],
                [ 0.0811, -0.0776, -0.0189, -0.1069,  0.0115, -0.0258, -0.0082,
                -0.0825],
                [-0.0244, -0.1143,  0.0299, -0.0557, -0.0132, -0.0258, -0.0557,
                    0.0488],
                [-0.0243, -0.1143,  0.0124, -0.1206, -0.0284,  0.0114, -0.0688,
                -0.0049]
            ]],
            dtype=activation_dtype,
            device=activation_device,
        )
    else:
        raise ValueError(f"Unknown key for toy test cases: {case_key}")

    # construct the input tensor
    torch.manual_seed(init_seed)
    input = torch.randn(b, s, h, dtype=activation_dtype, device=activation_device)
    
    # instantiate the module
    group_rms_norm = GroupRMSNorm(
        hidden_size=h,
        group_size=gz,
        eps=eps,
        init_range=init_range,
        init_seed=init_seed,
        dtype=param_dtype,
        device=param_device,
    )
    
    # apply the forward pass
    output = group_rms_norm(input)
    
    # check if the output tensor is correct
    assert_close(output, output_ref, atol=atol, rtol=rtol)
    

@pytest.mark.parametrize(
    "case_key, case_config",
    toy_test_cases["task2"].items(),
)
def test_task2(case_key, case_config):
    # set hyper parameters
    v, e = case_config["v"], case_config["e"]
    rank, world_size, process_group = case_config["rank"], case_config["world_size"], case_config.pop("process_group", None)
    init_mean, init_std = case_config.pop("init_mean"), case_config.pop("init_std") 
    init_base_seed = case_config.pop("init_base_seed", SEED)
    atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)
    param_dtype = case_config.pop("param_dtype", PARAM_DTYPE)
    activation_device, param_device = case_config["activation_device"], case_config.pop("param_device", PARAM_DEVICE)
    
    # construct the reference output embedding tensor
    if case_key == "case1":
        output_emb_ref = torch.tensor(
            [[[ 0.0000,  0.0000,  0.0000,  0.0000],
                [-0.0678, -0.0886, -0.3124, -0.3552],
                [ 0.0000,  0.0000,  0.0000,  0.0000],
                [-1.0999, -0.4687, -0.8400,  0.6868],
                [-0.5621, -0.4143, -1.3001, -0.1012],
                [ 0.0000,  0.0000,  0.0000,  0.0000]
            ]],
            dtype=param_dtype,
            device=activation_device,
        )
    else:
        raise ValueError(f"Unknown key for toy test cases: {case_key}")

    # construct the input ids
    input_ids = torch.tensor([[0, 11, 6, 10, 9, 5]], device=activation_device, dtype=torch.long)
    
    # instantiate the module
    parallel_vocab_emb = ParallelVocabEmbedding(
        vocab_size=v,
        emb_size=e,
        rank=rank,
        world_size=world_size,
        process_group=process_group,
        init_mean=init_mean,
        init_std=init_std,
        init_base_seed=init_base_seed,
        dtype=param_dtype,
        device=param_device,
    )
    
    # apply the forward pass
    output_emb = parallel_vocab_emb(input_ids)
    
    # check if the output embedding tensor is correct
    assert_close(output_emb, output_emb_ref, atol=atol, rtol=rtol)
    
    
@pytest.mark.parametrize(
    "case_key, case_config",
    toy_test_cases["task3"].items(),
)
def test_task3(case_key, case_config):
    # define hyper parameters
    b, s, nh, hd = case_config["b"], case_config["s"], case_config["nh"], case_config["hd"]
    ms, base, k = case_config["ms"], case_config["base"], case_config["k"]
    dynamic = case_config["dynamic"]
    atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)
    seed = case_config.pop("seed", SEED)
    activation_dtype, param_dtype = case_config["activation_dtype"], case_config.pop("param_dtype", PARAM_DTYPE)
    activation_device, param_device = case_config["activation_device"], case_config.pop("param_device", PARAM_DEVICE)

    # construct the reference output tensor
    if case_key == "case1":
        output_ref = torch.tensor(
            [[[[-0.8086, -1.5312,  0.4062,  0.1719]],
 
                [[-0.0835, -0.0737,  0.0145, -0.6484]],
        
                [[-0.6797,  0.2129,  0.3125,  1.1797]],
        
                [[ 1.3047, -0.6523, -0.2275, -0.6641]],
        
                [[ 1.6250,  1.3516,  1.4609, -0.0586]]
            ]],
            dtype=activation_dtype,
            device=activation_device,
        )
        new_ratio_ref = 1
    elif case_key == "case2":
        output_ref = torch.tensor(
            [[[[-0.8086, -1.5312,  0.4062,  0.1719]],
 
                [[ 0.6055,  0.2041, -0.6836, -0.3867]],
        
                [[ 0.5234,  0.2676,  0.9023, -0.2891]],
        
                [[-1.1875, -0.7812, -1.2188,  0.1167]],
        
                [[ 0.0596, -0.7617, -1.1953, -1.1875]],
        
                [[ 0.6172, -0.1152, -1.2422,  1.8359]],
        
                [[-0.2314, -0.2695, -0.5703, -0.2930]],
        
                [[-0.9375, -0.9141, -1.6094,  0.6562]],
        
                [[ 1.4219, -3.0781, -0.5117, -3.3281]]
            ]],
            dtype=activation_dtype,
            device=activation_device,
        )
        new_ratio_ref = 6
    else:
        raise ValueError(f"Unknown key for toy test cases: {case_key}")

    # construct the input tensor
    torch.manual_seed(seed)
    input = torch.randn(b, s, nh, hd, dtype=activation_dtype, device=activation_device)
    
    # instantiate the module
    ntk_rope = NTKAwareRoPE(
        dim=hd,
        max_seq_len=ms,
        base=base,
        ratio=k,
        dynamic=dynamic,
        dtype=param_dtype,
        device=param_device,
    )
    
    # apply the forward pass
    output = ntk_rope(input)
    
    # check if the output tensor is correct
    assert_close(output, output_ref, atol=atol, rtol=rtol)
    assert ntk_rope.ratio == new_ratio_ref, f"The current NTK-aware RoPE scaling ratio should be {new_ratio_ref}, but got {ntk_rope.ratio}"


if __name__ == "__main__":
    pytest.main()
    