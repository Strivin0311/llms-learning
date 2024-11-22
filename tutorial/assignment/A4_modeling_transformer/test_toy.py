import sys
sys.path.insert(0, ".")

from typing import List, Optional, Sequence, Dict, Any

import pytest

import torch
import torch.nn.functional as F
from torch.testing import assert_close

from src.modeling import (
    MLPActivationType,
    AttnQKVLayout,
    AttnQKVPackFormat,
    TransformerConfig,
    TransformerDecoderKVCache,
    TransformerDecoderLayer,
    TransformerDecoderBlock,
)

# constants for all toy test cases
ATOL = 1e-2
RTOL = 5e-2
SEED = 42
PARAM_DEVICE = "cpu"
PARAM_DTYPE = torch.float32

# configs for each toy test case
toy_test_cases = {
    "task1": {
        "case1": {
            "b": 1,
            "nh": 1,
            "hd": 4,
            "qkv_layout": AttnQKVLayout.BSHD,
            "num_layers": 2,
            "ops": [
                {
                    "op": "has",
                    "layer_idx": 0,
                },
                {
                    "op": "set",
                    "layer_idx": 1,
                    "s": 3,
                    "seqlens": None,
                },
                {
                    "op": "set",
                    "layer_idx": 0,
                    "s": 2,
                    "seqlens": None,
                },
                {
                    "op": "has",
                    "layer_idx": 1,
                },
                {
                    "op": "get",
                    "layer_idx": 1,
                },
                {
                    "op": "reset",
                },
                {
                    "op": "has",
                    "layer_idx": 1,
                },
                {
                    "op": "append",
                    "layer_idx": 0,
                    "s": 1,
                    "seqlens": None,
                },
                {
                    "op": "append",
                    "layer_idx": 0,
                    "s": 2,
                    "seqlens": None,
                },
                {
                    "op": "has",
                    "layer_idx": 0,
                },
                {
                    "op": "get",
                    "layer_idx": 0,
                },
            ]
        },
        "case2": {
            "b": 1,
            "nh": 1,
            "hd": 4,
            "qkv_layout": AttnQKVLayout.THD,
            "num_layers": 2,
            "ops": [
                {
                    "op": "has",
                    "layer_idx": 0,
                },
                {
                    "op": "set",
                    "layer_idx": 1,
                    "s": 5,
                    "seqlens": [2,2,1],
                },
                {
                    "op": "append",
                    "layer_idx": 0,
                    "s": 4,
                    "seqlens": [1,1,2],
                },
                {
                    "op": "has",
                    "layer_idx": 0,
                },
                {
                    "op": "get",
                    "layer_idx": 0,
                },
                {
                    "op": "append",
                    "layer_idx": 1,
                    "s": 3,
                    "seqlens": [1,1,1],
                },
                {
                    "op": "has",
                    "layer_idx": 1,
                },
                {
                    "op": "get",
                    "layer_idx": 1,
                },
            ]
        },
    },
    "task2": {
        "case1": {
            "training": True,
            
            "b": 1,
            "s": 8,
            "seqlens": None,
            "past_seqlen_kv": 0,
            "past_seqlens": None,
            "activation_dtype": torch.bfloat16,
            "activation_device": "cpu",
            "config": TransformerConfig(
                num_layers=1,
                hidden_size=8,
                ffh_size=16,
                max_seq_len=8,
                param_dtype=PARAM_DTYPE,
                param_device=PARAM_DEVICE,
                init_base_seed=SEED,
                
                vocab_size=10,
                vocab_init_mean=0.1,
                vocab_init_std=1.1,
                
                rope_base=10000,
                rope_ratio=1,
                rope_dynamic=False,
                
                group_size=None,
                eps=1e-5,
                norm_init_range=(-1.1, 1.1),
                
                proj_init_seed=SEED,
                proj_init_mean=0.1,
                proj_init_std=1.1,
                lm_head_tied=False,
                
                online_attn_block_size=4,
                head_dim=4,
                num_q_head=2,
                num_kv_head=1,
                qkv_pack_format=AttnQKVPackFormat.Q_K_V,
                qkv_layout=AttnQKVLayout.BSHD,
                window_size=None,
                causal=True,
                softmax_dropout_rate=0.,
                softmax_dropout_seed=SEED,
                softmax_scale=None,
                softmax_cap=None,
                softmax_temp=1.,
                softmax_clip_range=(0., 1.),
                apply_qk_norm=False,
                qk_norm_group_size=None,
                
                activation_type=MLPActivationType.SILU,
                lora_rank=0,
                lora_alpha=None,
                lora_dropout_rate=0.,
                lora_dropout_seed=SEED,
                lora_init_base_seed=SEED,
                
                num_experts=None,
                moe_topk=1,
                gate_init_mean=0.,
                gate_init_std=1.,
            ),
        },
        "case2": {
            "training": False,
            
            "b": 1,
            "s": 1,
            "seqlens": None,
            "past_seqlen_kv": 5,
            "past_seqlens": None,
            "activation_dtype": torch.bfloat16,
            "activation_device": "cpu",
            "config": TransformerConfig(
                num_layers=2,
                hidden_size=8,
                ffh_size=16,
                max_seq_len=16,
                param_dtype=PARAM_DTYPE,
                param_device=PARAM_DEVICE,
                init_base_seed=SEED,
                
                vocab_size=10,
                vocab_init_mean=0.1,
                vocab_init_std=1.1,
                
                rope_base=10000,
                rope_ratio=1,
                rope_dynamic=False,
                
                group_size=None,
                eps=1e-5,
                norm_init_range=(-1.1, 1.1),
                
                proj_init_seed=SEED,
                proj_init_mean=0.1,
                proj_init_std=1.1,
                lm_head_tied=False,
                
                online_attn_block_size=None,
                head_dim=4,
                num_q_head=2,
                num_kv_head=1,
                qkv_pack_format=AttnQKVPackFormat.Q_KV,
                qkv_layout=AttnQKVLayout.SBHD,
                window_size=None,
                causal=True,
                softmax_dropout_rate=0.,
                softmax_dropout_seed=SEED,
                softmax_scale=None,
                softmax_cap=None,
                softmax_temp=1.,
                softmax_clip_range=(0., 1.),
                apply_qk_norm=False,
                qk_norm_group_size=None,
                
                activation_type=MLPActivationType.SILU,
                lora_rank=0,
                lora_alpha=None,
                lora_dropout_rate=0.,
                lora_dropout_seed=SEED,
                lora_init_base_seed=SEED,
                
                num_experts=None,
                moe_topk=1,
                gate_init_mean=0.,
                gate_init_std=1.,
            ),
        },
        "case3": {
            "training": False,
            
            "b": 1,
            "s": 3,
            "seqlens": [1, 1, 1],
            "past_seqlen_kv": 12,
            "past_seqlens": [5, 4, 3],
            "activation_dtype": torch.bfloat16,
            "activation_device": "cpu",
            "config": TransformerConfig(
                num_layers=3,
                hidden_size=8,
                ffh_size=16,
                max_seq_len=16,
                param_dtype=PARAM_DTYPE,
                param_device=PARAM_DEVICE,
                init_base_seed=SEED,
                
                vocab_size=10,
                vocab_init_mean=0.1,
                vocab_init_std=1.1,
                
                rope_base=10000,
                rope_ratio=1,
                rope_dynamic=False,
                
                group_size=None,
                eps=1e-5,
                norm_init_range=(-1.1, 1.1),
                
                proj_init_seed=SEED,
                proj_init_mean=0.1,
                proj_init_std=1.1,
                lm_head_tied=False,
                
                online_attn_block_size=None,
                head_dim=4,
                num_q_head=2,
                num_kv_head=1,
                qkv_pack_format=AttnQKVPackFormat.Q_K_V,
                qkv_layout=AttnQKVLayout.THD,
                window_size=None,
                causal=True,
                softmax_dropout_rate=0.,
                softmax_dropout_seed=SEED,
                softmax_scale=None,
                softmax_cap=None,
                softmax_temp=1.,
                softmax_clip_range=(0., 1.),
                apply_qk_norm=False,
                qk_norm_group_size=None,
                
                activation_type=MLPActivationType.SILU,
                lora_rank=0,
                lora_alpha=None,
                lora_dropout_rate=0.,
                lora_dropout_seed=SEED,
                lora_init_base_seed=SEED,
                
                num_experts=None,
                moe_topk=1,
                gate_init_mean=0.,
                gate_init_std=1.,
            )
        }
    },
}
toy_test_cases["task3"] = toy_test_cases["task2"] # share the same test cases


def construct_kvcache_args(
    b: int,
    nh: int,
    hd: int,
    qkv_layout: AttnQKVLayout,
    ops: List[Dict[str, Any]],
    dtype: torch.dtype = PARAM_DTYPE,
    device: str = PARAM_DEVICE,
    seed: int = SEED,
) -> List[Sequence[Optional[torch.Tensor]]]:
    input_tensors = []
    
    for i, op in enumerate(ops):
        if op['op'] in ("set", "append"):
            s, seqlens = op['s'], op['seqlens']
            
            torch.manual_seed(seed + i)
            k = torch.randn(b, s, nh, hd, dtype=dtype, device=device)
            v = torch.randn_like(k)
            cu_seqlens = None
            
            if qkv_layout == AttnQKVLayout.BSHD:
                pass
            elif qkv_layout == AttnQKVLayout.SBHD:
                k, v = [x.transpose(0, 1) for x in (k, v)]
            elif qkv_layout == AttnQKVLayout.THD:
                assert b == 1, "b should be equal to 1 when qkv_layout is THD"
                assert seqlens is not None, "seqlens must be given when qkv_layout is THD"
                k, v = [x.squeeze(0) for x in (k, v)]
                cu_seqlens = torch.concat([
                    torch.zeros(1, dtype=torch.int32, device=device),
                    torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(dim=0)
                ], dim=0).to(torch.int32)
                assert cu_seqlens[-1] == (t := b * s), f"The sum of seqlens ({cu_seqlens[-1]}) != length ({t})"
            else:
                raise ValueError(f"Unsupported qkv_layout: {qkv_layout}")

            input_tensors.append((k, v, cu_seqlens))
        else:
            input_tensors.append(None)
        
    return input_tensors
           

def construct_decoder_args(
    config: TransformerConfig,
    b: int,
    s: int,
    seqlens: Optional[List[int]] = None,
    past_seqlen_kv: int = 0,
    past_seqlens: Optional[List[int]] = None,
    dtype: torch.dtype = PARAM_DTYPE,
    device: str = PARAM_DEVICE,
) -> Sequence[Optional[torch.Tensor]]:
    torch.manual_seed(config.init_base_seed)
    input = torch.randn(b, s, config.hidden_size, dtype=dtype, device=device)
    input_ids = torch.randint(0, config.vocab_size, (b, s), dtype=torch.int32, device=device)
    
    if seqlens is not None:
        assert config.qkv_layout is AttnQKVLayout.THD, "if using varlen attn, the qkv_layout must be THD"
        assert b == 1, "b should be equal to 1 if using varlen attn"
        
        cu_seqlens = torch.concat([
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(dim=0)
        ], dim=0).to(torch.int32)
        assert cu_seqlens[-1] == (t:=b*s), f"The sum of seqlens ({cu_seqlens[-1]}) != b*s ({t})"
    else:
        cu_seqlens = None
    
    if past_seqlen_kv > 0:
        if config.qkv_layout is AttnQKVLayout.THD:
            assert past_seqlens is not None, "past_seqlens must be given when qkv_layout is THD and past_seqlen_kv > 0"
        kv_cache = TransformerDecoderKVCache(
            qkv_layout=config.qkv_layout,
            num_layers=config.num_layers,
        )
        
        for layer_idx in range(config.num_layers):
            torch.manual_seed(config.init_base_seed + layer_idx)
            past_k = torch.randn(
                b, past_seqlen_kv, config.num_kv_head, config.head_dim, 
                dtype=config.param_dtype, device=config.param_device
            )
            past_v = torch.randn_like(past_k)
            past_cu_seqlens = None
            
            if config.qkv_layout == AttnQKVLayout.BSHD:
                pass
            elif config.qkv_layout == AttnQKVLayout.SBHD:
                past_k, past_v = [x.transpose(0, 1) for x in (past_k, past_v)]
            elif config.qkv_layout == AttnQKVLayout.THD:
                past_k, past_v = [x.squeeze(0) for x in (past_k, past_v)]
                past_cu_seqlens = torch.concat([
                    torch.zeros(1, dtype=torch.int32, device=device),
                    torch.tensor(past_seqlens, dtype=torch.int32, device=device).cumsum(dim=0)
                ], dim=0).to(torch.int32)
                assert past_cu_seqlens[-1] == (t := len(past_k)), \
                    f"The sum of past seqlens ({past_cu_seqlens[-1]}) != past length ({t})"
            else:
                raise ValueError(f"Unsupported qkv_layout: {config.qkv_layout}")
            
            kv_cache.set(layer_idx, past_k, past_v, cu_seqlens=past_cu_seqlens)
    else:
        kv_cache = None
    
    return input, input_ids, cu_seqlens, kv_cache


@pytest.mark.parametrize(
    "case_key, case_config",
    toy_test_cases["task1"].items(),
)
def test_task1(case_key, case_config):
    # set hyper parameters
    b, nh, hd = case_config["b"], case_config["nh"], case_config["hd"]
    qkv_layout, num_layers = case_config["qkv_layout"], case_config["num_layers"]
    ops = case_config["ops"]
    atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)
    kv_dtype, kv_device = case_config.pop("kv_dtype", PARAM_DTYPE), case_config.pop("kv_device", PARAM_DEVICE)
    seed = case_config.pop("seed", SEED)
    
    # construct the reference output tensors
    if case_key == "case1":
        outputs_ref = [
            False,
            None,
            None,
            True,
            (
                torch.tensor(
                    [[[[-0.6484, -0.7058,  0.6432,  1.4788]],
            
                    [[ 1.1918, -0.1446,  0.4847,  0.6921]],
            
                    [[-1.3929,  0.7623,  0.8387, -1.0450]]]],
                    dtype=kv_dtype,
                    device=kv_device,
                ),
                torch.tensor(
                    [[[[ 1.1097,  0.3953,  1.1804, -0.8989]],
                
                    [[-0.8313,  0.4680,  2.2700,  0.0743]],
            
                    [[-0.8931, -0.9201, -0.0213,  1.7711]]]],
                    dtype=kv_dtype,
                    device=kv_device,
                    ),
                None,
            ),
            None,
            False,
            None,
            None,
            True,
            (
                torch.tensor(
                    [[[[-2.0157,  2.0106,  0.0583,  0.0656]],
            
                    [[ 0.4625, -0.1692,  0.3719,  1.4709]],
            
                    [[-0.1568, -2.8720,  1.9054, -0.1457]]]],
                    dtype=kv_dtype,
                    device=kv_device,
                ),
                torch.tensor(
                    [[[[-1.6534,  2.2517,  0.9501,  2.2385]],
            
                    [[-1.8826, -1.0217, -0.2169, -1.0115]],
            
                    [[ 0.1614, -0.0939,  1.7723, -0.0284]]]],
                    dtype=kv_dtype,
                    device=kv_device,
                ),
                None
            )
        ]
    elif case_key == "case2":
        outputs_ref = [
            False,
            None,
            None,
            True,
            (
                torch.tensor(
                    [[[ 1.5862,  1.1253,  1.8306,  0.1129]],
            
                    [[ 0.4976,  1.5010, -0.1413, -0.3522]],
            
                    [[-0.1643, -1.1651, -0.4089, -0.5252]],
            
                    [[-1.3153,  0.6031, -0.8124,  0.5920]]],
                    dtype=kv_dtype,
                    device=kv_device,
                ),
                torch.tensor(
                    [[[-1.2266, -0.9598,  1.7118, -0.0146]],
            
                    [[ 0.4252, -1.3446,  1.6114,  0.5914]],
            
                    [[ 0.1644,  1.2514,  0.5173, -0.8078]],
            
                    [[-2.0788,  0.6370,  1.3824, -0.9156]]],
                    dtype=kv_dtype,
                    device=kv_device,
                ),
                torch.tensor([0, 1, 2, 4], dtype=torch.int32, device=kv_device),
            ),
            None,
            True,
            (
                torch.tensor(
                    [[[-0.0166, -0.4668,  2.0909,  0.6149]],
            
                    [[ 0.3083, -0.2947, -0.7662, -0.9962]],
            
                    [[-1.4624,  0.7523, -1.7173,  0.5757]],
            
                    [[-0.2345, -0.5367,  1.1296,  0.1054]],
            
                    [[-0.3630,  1.5822, -0.4430,  1.8462]],
            
                    [[ 0.6040,  1.1914,  0.3525,  0.2941]],
            
                    [[-0.4772, -1.8291, -0.6145,  1.0282]],
            
                    [[ 0.5197, -0.1634, -0.0875,  0.6146]]],
                    dtype=kv_dtype,
                    device=kv_device,
                ),
                torch.tensor(
                    [[[-0.7771, -0.4484, -1.1668,  0.5006]],
            
                    [[ 0.0139,  0.6564,  0.4846, -0.2549]],
            
                    [[ 0.3034,  0.7770, -2.0360,  0.3562]],
            
                    [[-0.7603, -1.6943, -0.2596,  0.8847]],
            
                    [[-0.8256,  0.7988, -0.3005, -0.3062]],
            
                    [[ 0.8027, -0.6474,  0.4054, -0.5901]],
            
                    [[ 0.4163, -0.5947, -0.2367, -1.8343]],
            
                    [[-0.5833, -0.6801, -1.2947,  0.9606]]],
                    dtype=kv_dtype,
                    device=kv_device,
                ),
                torch.tensor([0, 3, 6, 8], dtype=torch.int32, device=kv_device),
            )
        ]
    else:
        raise ValueError(f"Unknown key for toy test cases: {case_key}")
        
    # construct the input tensors
    input_tensors = construct_kvcache_args(
        b, nh, hd, qkv_layout, 
        ops,
        kv_dtype, kv_device,
        seed,
    )
    
    # instantiate the module
    kv_cache = TransformerDecoderKVCache(
        qkv_layout=qkv_layout,
        num_layers=num_layers,
    )
    
    # apply each operation to check if the output for each operation is correct
    for i, (op, input_tensor, output_ref) in enumerate(zip(ops, input_tensors, outputs_ref)):
        try:
            if op['op'] == "reset":
                kv_cache.reset()
            elif op['op'] == "has":
                layer_idx = op['layer_idx']
                assert kv_cache.has(layer_idx) == output_ref
            elif op['op'] == "get":
                layer_idx = op['layer_idx']
                k, v, cu_seqlens = kv_cache.get(layer_idx)
                k_ref, v_ref, cu_seqlens_ref = output_ref
                assert_close(k, k_ref, atol=atol, rtol=rtol)
                assert_close(v, v_ref, atol=atol, rtol=rtol)
                if cu_seqlens_ref is not None:
                    assert_close(cu_seqlens, cu_seqlens_ref, atol=atol, rtol=rtol)
                else:
                    assert cu_seqlens is None
            elif op['op'] == "set":
                layer_idx = op['layer_idx']
                k, v, cu_seqlens = input_tensor
                kv_cache.set(layer_idx, k, v, cu_seqlens=cu_seqlens)
            elif op['op'] == "append":
                layer_idx = op['layer_idx']
                k, v, cu_seqlens = input_tensor
                kv_cache.append(layer_idx, k, v, cu_seqlens=cu_seqlens)
            else:
                raise ValueError(f"Unknown operation: {op['op']}")

        except Exception as e:
            assert False, f"The {i+1}-th operation `{op['op']}` failed due to the error: {e}"
        

@pytest.mark.parametrize(
    "case_key, case_config",
    toy_test_cases["task2"].items(),
)
def test_task2(case_key, case_config):
    # set hyper parameters
    b, s, seqlens = case_config["b"], case_config["s"], case_config["seqlens"]
    past_seqlen_kv, past_seqlens = case_config["past_seqlen_kv"], case_config["past_seqlens"]
    config: TransformerConfig = case_config["config"]
    activation_dtype, activation_device = case_config["activation_dtype"], case_config["activation_device"]
    atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)
    if config.qkv_layout is AttnQKVLayout.THD:
        assert seqlens is not None, "seqlens must be given when qkv_layout is THD"
        if past_seqlen_kv > 0:
            assert past_seqlens is not None, "past_seqlens must be given when qkv_layout is THD and past_seqlen_kv > 0"
    
    # construct the reference output tensor
    if case_key == "case1":
        output_ref = torch.tensor(
            [[[ -1.6328, -14.3125,  -4.7188,  -5.0938,   0.3770,  -1.6016,  -5.3750,
                0.8789],
            [  5.0312,   9.8750,  -2.1562,   5.3125,  -2.2188,   5.2500,   0.2373,
                -6.2188],
            [  2.4844,  -9.8750,   0.8711,  -5.2500,   4.5625,  -6.5625,  -2.1875,
                -0.0664],
            [ -3.4531, -15.8750,   3.1250,  -7.9688,   1.1484, -12.6875,   1.8281,
                5.1562],
            [  4.4062,   0.1680,   0.3398,  -2.5938,   7.0938,   0.2715,  -8.1875,
                7.6562],
            [ -3.8594, -16.3750,   2.7031,  -6.7812,   1.6875, -15.1875,   3.3594,
                7.3438],
            [ -3.4688,   1.3750,  -1.3125,   0.7305,  -5.9062,  -0.9570,   2.7500,
                0.0913],
            [ -9.1250,   0.3516,   2.3438,   3.2656, -13.2500,  -4.8750,   8.5625,
                6.7812]]
            ],
            dtype=activation_dtype,
            device=activation_device,
        )
    elif case_key == "case2":
        output_ref = torch.tensor(
            [[[ 2.1250, -6.6875,  3.5625, -2.6719,  3.8750, -7.8438,  2.1875, -0.7852]]],
            dtype=activation_dtype,
            device=activation_device,
        )
    elif case_key == "case3":
        output_ref = torch.tensor(
            [[[-1.8906, -9.5625, -4.8125, -1.9688,  4.5312, -2.0938, -4.7812,
                0.2119],
            [ 0.5859,  5.4062, -0.5000,  3.5938, -1.8828,  1.2734,  1.0469,
            -3.1250],
            [-6.5000, -8.9375,  4.3750, -0.1699, 11.8125, -7.8750,  0.5078,
                2.1250]]
            ],
            dtype=activation_dtype,
            device=activation_device,
        )
    else:
        raise ValueError(f"Unknown key for toy test cases: {case_key}")

    # construct the input tensor
    input, _, cu_seqlens, kv_cache = construct_decoder_args(
        config, 
        b, s, seqlens,
        past_seqlen_kv, past_seqlens,
        dtype=activation_dtype,
        device=activation_device,
    )
    
    # instantiate the module
    layer = TransformerDecoderLayer(config)
    
    # apply the forward pass
    output = layer(input, cu_seqlens=cu_seqlens, kv_cache=kv_cache)
    
    # check if the output tensor is correct
    assert_close(output, output_ref, atol=atol, rtol=rtol)
    

@pytest.mark.parametrize(
    "case_key, case_config",
    toy_test_cases["task3"].items(),
)
def test_task3(case_key, case_config):
    # set hyper parameters
    b, s, seqlens = case_config["b"], case_config["s"], case_config["seqlens"]
    past_seqlen_kv, past_seqlens = case_config["past_seqlen_kv"], case_config["past_seqlens"]
    config: TransformerConfig = case_config["config"]
    activation_dtype, activation_device = case_config["activation_dtype"], case_config["activation_device"]
    training = case_config.pop("training", True)
    assert training or config.online_attn_block_size is None, "online_attn_block_size must be None when training is False"
    atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)
    if config.qkv_layout is AttnQKVLayout.THD:
        assert seqlens is not None, "seqlens must be given when qkv_layout is THD"
        if past_seqlen_kv > 0:
            assert past_seqlens is not None, "past_seqlens must be given when qkv_layout is THD and past_seqlen_kv > 0"
    
    # construct the reference output tensor
    if case_key == "case1":
        logits_ref = torch.tensor(
            [[[-1.2925,  2.5357, -0.7369, -0.7364, -0.5004,  0.9618, -0.5549,
                1.1369, -0.5655,  0.4905],
            [-3.5297,  1.0178, -1.6529, -0.7515,  1.9548,  1.2376,  0.8119,
                1.6100, -1.3717,  0.7523],
            [-2.2095,  2.9566, -0.3049,  0.9765,  0.0953,  1.5885, -0.3909,
                2.4821,  0.5963,  1.2973],
            [ 4.7399,  2.9677, -1.3212,  1.0636, -2.2609,  2.0457, -0.8680,
            -1.5315,  0.4904,  2.3503],
            [-1.1233,  3.1436, -2.7241, -0.1384,  0.7005,  2.7857,  0.3798,
                1.8135, -0.5760,  2.2896],
            [-1.0801,  3.1166, -2.7623, -0.1017,  0.7399,  2.8142,  0.4089,
                1.7805, -0.5800,  2.3364],
            [-5.6624,  0.0164, -1.2714, -0.9070,  3.0396,  1.0790,  1.4420,
                2.4621, -1.7371,  0.3789],
            [ 1.4993,  4.2255, -0.8165,  2.0971, -0.9501,  2.6483, -1.1022,
                1.1436,  0.8783,  1.8043]]
            ],
            dtype=config.param_dtype,
            device=activation_device,
        )
    elif case_key == "case2":
        logits_ref = torch.tensor(
            [[[-0.0713, -0.7954,  1.0491, -4.1084, -1.7625, -1.1286, -0.6247,
           -1.1881, -2.9059, -3.9017]]],
            dtype=config.param_dtype,
            device=activation_device,
        )
    elif case_key == "case3":
        logits_ref = torch.tensor(
            [[[-5.7833e+00, -4.1559e+00, -7.2531e-01, -1.2458e+00,  5.3105e+00,
            -1.1368e+00,  2.0874e+00,  3.9418e-01, -3.1351e+00, -1.5997e+00],
            [ 5.1554e-01, -2.8597e+00,  7.3401e-01,  7.7610e-01,  5.3917e-01,
            -1.0834e+00,  1.0781e+00, -7.7173e-01,  1.2294e+00,  1.4850e-01],
            [ 8.8537e-01,  1.7028e+00, -5.3950e-03,  4.7097e-01, -1.2100e+00,
                1.1996e+00, -4.4219e-01,  2.1394e-01, -2.2099e-02,  1.5710e+00]]
            ],
            dtype=config.param_dtype,
            device=activation_device,
        )
    else:
        raise ValueError(f"Unknown key for toy test cases: {case_key}")

    # construct the input tensors
    _, input_ids, cu_seqlens, kv_cache = construct_decoder_args(
        config,
        b, s, seqlens,
        past_seqlen_kv, past_seqlens,
        dtype=activation_dtype,
        device=activation_device,
    )
    
    # instantiate the module
    block = TransformerDecoderBlock(config)
    block = block.train() if training else block.eval()
    if kv_cache is not None:
        block.set_kv_cache(kv_cache)
    
    # apply the forward pass
    logits = block(input_ids, cu_seqlens=cu_seqlens)
    
    # check if the output logits tensor is correct
    assert_close(logits, logits_ref, atol=atol, rtol=rtol)
 

if __name__ == "__main__":
    pytest.main()
    