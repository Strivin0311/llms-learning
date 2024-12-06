import sys
sys.path.insert(0, ".")

import pytest

import torch
from torch.testing import assert_close

from ref.modeling import (
    GroupRMSNorm as GroupRMSNormRef,
    ParallelVocabEmbedding as ParallelVocabEmbeddingRef,
    NTKAwareRoPE as NTKAwareRoPERef,
)

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

    # construct the input tensor
    torch.manual_seed(init_seed)
    input = torch.randn(b, s, h, dtype=activation_dtype, device=activation_device)
    
    # instantiate the reference module
    group_rms_norm_ref = GroupRMSNormRef(
        hidden_size=h,
        group_size=gz,
        eps=eps,
        init_range=init_range,
        init_seed=init_seed,
        dtype=param_dtype,
        device=param_device,
    )
    
    # apply the forward pass to get the reference output tensor
    output_ref = group_rms_norm_ref(input.clone())
    
    # instantiate the student's module
    group_rms_norm = GroupRMSNorm(
        hidden_size=h,
        group_size=gz,
        eps=eps,
        init_range=init_range,
        init_seed=init_seed,
        dtype=param_dtype,
        device=param_device,
    )
    
    # apply the forward pass to get student's output tensor
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

    # construct the input ids
    input_ids = torch.tensor([[0, 11, 6, 10, 9, 5]], device=activation_device, dtype=torch.long)
    
    # instantiate the reference module
    parallel_vocab_emb_ref = ParallelVocabEmbeddingRef(
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
    
    # apply the forward pass to get the reference output embedding tensor
    output_emb_ref = parallel_vocab_emb_ref(input_ids.clone())
    
    # instantiate the student's module
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
    
    # apply the forward pass to get student's output embedding tensor
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

    # construct the input tensor
    torch.manual_seed(seed)
    input = torch.randn(b, s, nh, hd, dtype=activation_dtype, device=activation_device)
    
    # instantiate the reference module
    ntk_rope_ref = NTKAwareRoPERef(
        dim=hd,
        max_seq_len=ms,
        base=base,
        ratio=k,
        dynamic=dynamic,
        dtype=param_dtype,
        device=param_device,
    )
    
    # apply the forward pass to get the reference output embedding tensor
    output_ref = ntk_rope_ref(input.clone())
    new_ratio_ref = ntk_rope_ref.ratio
    
    # instantiate the student's module
    ntk_rope = NTKAwareRoPE(
        dim=hd,
        max_seq_len=ms,
        base=base,
        ratio=k,
        dynamic=dynamic,
        dtype=param_dtype,
        device=param_device,
    )
    
    # apply the forward pass to get student's output embedding tensor
    output = ntk_rope(input)
    
    # check if the output tensor is correct
    assert_close(output, output_ref, atol=atol, rtol=rtol)
    assert ntk_rope.ratio == new_ratio_ref, f"The current NTK-aware RoPE scaling ratio should be {new_ratio_ref}, but got {ntk_rope.ratio}"


if __name__ == "__main__":
    pytest.main()
    