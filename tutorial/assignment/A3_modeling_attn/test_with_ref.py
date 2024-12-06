import sys
sys.path.insert(0, ".")

from typing import List, Optional, Sequence

import pytest

import torch
import torch.nn.functional as F
from torch.testing import assert_close

from ref.modeling import (
    AttnQKVLayout as AttnQKVLayoutRef,
    AttnQKVPackFormat as AttnQKVPackFormatRef,
    OfflineSlidingWindowAttn as OfflineSlidingWindowAttnRef,
    OnlineSlidingWindowAttn as OnlineSlidingWindowAttnRef,
)

from src.modeling import (
    AttnQKVPackFormat,
    AttnQKVLayout,
    OfflineSlidingWindowAttn,
    OnlineSlidingWindowAttn,
)

# constants for all toy test cases
ATOL = 1e-3
RTOL = 1e-3
SEED = 42
PARAM_DEVICE = "cpu"
PARAM_DTYPE = torch.float32

# mapping from ref attn_qkv_layout to student attn_qkv_layout
attn_qkv_layout_ref_to_student = {
    AttnQKVLayoutRef.BSHD: AttnQKVLayout.BSHD,
    AttnQKVLayoutRef.SBHD: AttnQKVLayout.SBHD,
    AttnQKVLayoutRef.THD: AttnQKVLayout.THD,
}

# mapping from ref attn_qkv_pack_format to student attn_qkv_pack_format
attn_qkv_pack_format_ref_to_student = {
    AttnQKVPackFormatRef.QKV: AttnQKVPackFormat.QKV,
    AttnQKVPackFormatRef.Q_KV: AttnQKVPackFormat.Q_KV,
    AttnQKVPackFormatRef.Q_K_V: AttnQKVPackFormat.Q_K_V,
}

# configs for each toy test case
toy_test_cases = {
    "task1": {
        "case1": {
            "b": 1,
            "sq": 6,
            "skv": 6,
            "hq": 1,
            "hkv": 1,
            "hd": 4,
            
            "qkv_pack_format": AttnQKVPackFormatRef.QKV,
            "qkv_layout": AttnQKVLayoutRef.SBHD,
            
            "seqlens_q": None,
            "seqlens_kv": None,
            
            "window_size": None,
            "causal": True,
            
            "softmax_dropout_rate": 0.1,
            "softmax_scale": None,
            "softmax_cap": None,
            "softmax_temp": 0.8,
            "softmax_clip_range": (-0.03, 1.03),
            
            "apply_qk_norm": True,
            "group_size": 1,
            "eps": 1e-5,
            "init_range": (-1.1, 1.1),
            
            "activation_dtype": torch.bfloat16,
            "activation_device": "cpu",
        },
        "case2": {
            "b": 1,
            "sq": 7,
            "skv": 5,
            "hq": 2,
            "hkv": 1,
            "hd": 4,
            
            "qkv_pack_format": AttnQKVPackFormatRef.Q_KV,
            "qkv_layout": AttnQKVLayoutRef.THD,
            
            "seqlens_q": [1, 2, 4],
            "seqlens_kv": [2, 2, 1],
            
            "window_size": 1,
            "causal": False,
            
            "softmax_dropout_rate": 0.0,
            "softmax_scale": None,
            "softmax_cap": 10.0,
            "softmax_temp": 1.0,
            "softmax_clip_range": (-0.01, 1.01),
            
            "apply_qk_norm": True,
            "group_size": 2,
            "eps": 1e-5,
            "init_range": (-1.2, 1.2),
            
            "activation_dtype": torch.float32,
            "activation_device": "cpu",
        }
    },
    "task2": {
        "case1": {
            "b": 1,
            "sq": 7,
            "skv": 5,
            "hq": 1,
            "hkv": 1,
            "hd": 4,
            
            "bq": 3,
            "bkv": 2,
            "bqi": 1,
            "bkvj": 1,
            
            "window_size": 2,
            "causal": True,
            
            "softmax_scale": None,
            "softmax_cap": 10.0,
            "softmax_temp": 1.0,
            
            "apply_qk_norm": True,
            "group_size": 2,
            "eps": 1e-5,
            "init_range": (-1.05, 1.05),
            
            "activation_dtype": torch.float32,
            "activation_device": "cpu",
        },
        "case2": {
            "b": 1,
            "sq": 7,
            "skv": 5,
            "hq": 1,
            "hkv": 1,
            "hd": 4,
            
            "bq": 3,
            "bkv": 2,
            "bqi": 2,
            "bkvj": 0,
            
            "window_size": 1,
            "causal": False,
            
            "softmax_scale": None,
            "softmax_cap": None,
            "softmax_temp": 0.9,
            
            "apply_qk_norm": True,
            "group_size": 1,
            "eps": 1e-5,
            "init_range": (-1.25, 1.25),
            
            "activation_dtype": torch.float32,
            "activation_device": "cpu",
        }
    },
}


def construct_offline_attn_args(
    b: int,
    sq: int,
    skv: int,
    hq: int,
    hkv: int,
    hd: int,
    qkv_pack_format: AttnQKVPackFormatRef,
    qkv_layout: AttnQKVLayoutRef,
    seqlens_q: Optional[List[int]] = None,
    seqlens_kv: Optional[List[int]] = None,
    dtype: torch.dtype = PARAM_DTYPE,
    device: str = PARAM_DEVICE,
    seed: int = SEED,
) -> Sequence[Optional[torch.Tensor]]:
    torch.manual_seed(seed)
    q = torch.randn((b, sq, hq, hd), dtype=dtype, device=device)
    k = torch.randn((b, skv, hkv, hd), dtype=dtype, device=device)
    v = torch.randn((b, skv, hkv, hd), dtype=dtype, device=device)
    
    if qkv_layout == AttnQKVLayoutRef.THD:
        assert seqlens_q is not None, "THD layout requires cu_seqlens_q"
        assert seqlens_kv is not None, "THD layout requires cu_seqlens_kv"
        
        cu_seqlens_q, cu_seqlens_kv =[
            torch.concat([
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.tensor(x, dtype=torch.int32, device=device).cumsum(dim=0)
            ], dim=0)
            for x in (seqlens_q, seqlens_kv)
        ]
        
        assert cu_seqlens_q[-1] == b*sq, f"cu_seqlens_q[-1]({cu_seqlens_q[-1]}) == b*sq({b*sq})"
        assert cu_seqlens_kv[-1] == b*skv, f"cu_seqlens_kv[-1]({cu_seqlens_kv[-1]}) == b*skv({b*skv})"
        
        q, k, v = [
            x.view(-1, *x.shape[-2:]).contiguous() 
            for x in (q, k, v)
        ]
    else:
        assert seqlens_q is None, "QKV layout does not require cu_seqlens_q"
        assert seqlens_kv is None, "QKV layout does not require cu_seqlens_kv"
        cu_seqlens_q, cu_seqlens_kv = None, None
        
        if qkv_layout == AttnQKVLayoutRef.SBHD:
            q, k, v = [
                x.transpose(0, 1).contiguous() 
                for x in (q, k, v)
            ]
    
    if qkv_pack_format == AttnQKVPackFormatRef.QKV:
        assert sq == skv, "QKV pack format requires sq == skv"
        q = torch.concat((q, k, v), dim=-2)
        k, v = None, None
    elif qkv_pack_format == AttnQKVPackFormatRef.Q_KV:
        k = torch.concat((k, v), dim=-2)
        v = None
    
    return q, k, v, cu_seqlens_q, cu_seqlens_kv


def construct_online_attn_args(
    b: int,
    sq: int,
    skv: int,
    hq: int,
    hkv: int,
    hd: int,
    bq: int,
    bkv: int,
    bqi: int,
    bkvj: int,
    dtype: torch.dtype = PARAM_DTYPE,
    device: str = PARAM_DEVICE,
    seed: int = SEED,
) -> Sequence[torch.Tensor]:
    nbq = (sq + bq - 1) // bq
    nbk = (skv + bkv - 1) // bkv
    assert bqi < nbq, f"bqi({bqi}) >= nbq({nbq})"
    assert bkvj < nbk, f"bkvj({bkvj}) >= nbk({nbk})"
    
    torch.manual_seed(seed)
    q = torch.randn((b, sq, hq, hd), dtype=dtype, device=device)
    k = torch.randn((b, skv, hkv, hd), dtype=dtype, device=device)
    v = torch.randn((b, skv, hkv, hd), dtype=dtype, device=device)
    global_o = torch.randn_like(q)
    global_lse = torch.rand((b, hq, sq), dtype=torch.float32, device=device)
    
    q = F.pad(q, pad=(0, 0, 0, 0, 0, nbq*bq - sq), mode="constant", value=0)
    k = F.pad(k, pad=(0, 0, 0, 0, 0, nbk*bkv - skv), mode="constant", value=0)
    v = F.pad(v, pad=(0, 0, 0, 0, 0, nbk*bkv - skv), mode="constant", value=0)
    
    q = q[:, bqi*bq:(bqi+1)*bq, :, :]
    k = k[:, bkvj*bkv:(bkvj+1)*bkv, :, :]
    v = v[:, bkvj*bkv:(bkvj+1)*bkv, :, :]
    
    return q, k, v, global_o, global_lse


def safe_clone(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    else:
        return x.clone()


@pytest.mark.parametrize(
    "case_key, case_config",
    toy_test_cases["task1"].items(),
)
def test_task1(case_key, case_config):
    # set hyper parameters
    b, sq, skv = case_config["b"], case_config["sq"], case_config["skv"], 
    hq, hkv, hd = case_config["hq"], case_config["hkv"], case_config["hd"]
    qkv_pack_format, qkv_layout = case_config["qkv_pack_format"], case_config["qkv_layout"]
    seqlens_q, seqlens_kv = case_config["seqlens_q"], case_config["seqlens_kv"]
    window_size, causal = case_config["window_size"], case_config["causal"]
    softmax_dropout_rate, softmax_dropout_seed = case_config["softmax_dropout_rate"], \
        case_config.pop("softmax_dropout_seed", SEED)
    softmax_scale, softmax_cap, softmax_temp, softmax_clip_range = case_config["softmax_scale"], \
        case_config["softmax_cap"], case_config["softmax_temp"], case_config["softmax_clip_range"]
    apply_qk_norm, group_size, eps = case_config["apply_qk_norm"], case_config["group_size"], case_config["eps"]
    init_range, init_seed = case_config["init_range"], case_config.pop("init_seed", SEED)
    atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)
    activation_dtype, param_dtype = case_config["activation_dtype"], case_config.pop("param_dtype", PARAM_DTYPE)
    activation_device, param_device = case_config["activation_device"], case_config.pop("param_device", PARAM_DEVICE)

    # construct the input tensors
    q, k, v, cu_seqlens_q, cu_seqlens_kv = construct_offline_attn_args(
        b, sq, skv, hq, hkv, hd,
        qkv_pack_format, qkv_layout, seqlens_q, seqlens_kv, 
        dtype=activation_dtype, device=activation_device, seed=SEED,
    )
    
    # instantiate the reference module
    off_swa_ref = OfflineSlidingWindowAttnRef(
        head_dim=hd,
        num_q_head=hq,
        num_kv_head=hkv,
        qkv_pack_format=qkv_pack_format,
        qkv_layout=qkv_layout,
        window_size=window_size,
        causal=causal,
        softmax_dropout_rate=softmax_dropout_rate,
        softmax_dropout_seed=softmax_dropout_seed,
        softmax_scale=softmax_scale,
        softmax_cap=softmax_cap,
        softmax_temp=softmax_temp,
        softmax_clip_range=softmax_clip_range,
        apply_qk_norm=apply_qk_norm,
        group_size=group_size,
        eps=eps,
        init_range=init_range,
        init_seed=init_seed,
        dtype=param_dtype,
        device=param_device,
    )
    
    # apply the forward pass to get the reference output tensor
    output_ref = off_swa_ref(
        safe_clone(q), safe_clone(k), safe_clone(v),
        cu_seqlens_q=safe_clone(cu_seqlens_q),
        cu_seqlens_k=safe_clone(cu_seqlens_kv),
    )
    
    # instantiate the student's module
    off_swa = OfflineSlidingWindowAttn(
        head_dim=hd,
        num_q_head=hq,
        num_kv_head=hkv,
        qkv_pack_format=attn_qkv_pack_format_ref_to_student[qkv_pack_format],
        qkv_layout=attn_qkv_layout_ref_to_student[qkv_layout],
        window_size=window_size,
        causal=causal,
        softmax_dropout_rate=softmax_dropout_rate,
        softmax_dropout_seed=softmax_dropout_seed,
        softmax_scale=softmax_scale,
        softmax_cap=softmax_cap,
        softmax_temp=softmax_temp,
        softmax_clip_range=softmax_clip_range,
        apply_qk_norm=apply_qk_norm,
        group_size=group_size,
        eps=eps,
        init_range=init_range,
        init_seed=init_seed,
        dtype=param_dtype,
        device=param_device,
    )
    
    # apply the forward pass to get student's output tensor
    output = off_swa(
        q, k, v,
        cu_seqlens_q=cu_seqlens_q,
        cu_seqlens_k=cu_seqlens_kv,
    )
    
    # check if the output tensor is correct
    assert_close(output, output_ref, atol=atol, rtol=rtol)
    

@pytest.mark.parametrize(
    "case_key, case_config",
    toy_test_cases["task2"].items(),
)
def test_task2(case_key, case_config):
    # set hyper parameters
    b, sq, skv = case_config["b"], case_config["sq"], case_config["skv"], 
    hq, hkv, hd = case_config["hq"], case_config["hkv"], case_config["hd"]
    bq, bkv, bqi, bkvj = case_config["bq"], case_config["bkv"], case_config["bqi"], case_config["bkvj"]
    window_size, causal = case_config["window_size"], case_config["causal"]
    softmax_scale, softmax_cap, softmax_temp = case_config["softmax_scale"], \
        case_config["softmax_cap"], case_config["softmax_temp"]
    apply_qk_norm, group_size, eps = case_config["apply_qk_norm"], case_config["group_size"], case_config["eps"]
    init_range, init_seed = case_config["init_range"], case_config.pop("init_seed", SEED)
    atol, rtol = case_config.pop("atol", ATOL), case_config.pop("rtol", RTOL)
    activation_dtype, param_dtype = case_config["activation_dtype"], case_config.pop("param_dtype", PARAM_DTYPE)
    activation_device, param_device = case_config["activation_device"], case_config.pop("param_device", PARAM_DEVICE)

    # construct the input tensors
    q, k, v, global_o, global_lse = construct_online_attn_args(
        b, sq, skv, hq, hkv, hd, bq, bkv, bqi, bkvj,
        dtype=activation_dtype, device=activation_device, seed=SEED,
    )
    global_o_ref, global_lse_ref = global_o.clone(), global_lse.clone()
    
    # instantiate the reference module
    on_swa_ref = OnlineSlidingWindowAttnRef(
        seqlen_q=sq,
        seqlen_kv=skv,
        block_size_q=bq,
        block_size_kv=bkv,
        head_dim=hd,
        num_q_head=hq,
        num_kv_head=hkv,
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
        dtype=param_dtype,
        device=param_device,
    )
    
    # apply the forward pass to get the reference output tensors
    on_swa_ref(
        q.clone(), k.clone(), v.clone(),
        global_o=global_o_ref,
        global_lse=global_lse_ref,
        block_idx_q=bqi, 
        block_idx_kv=bkvj,
    )
    
    # instantiate the student's module
    on_swa = OnlineSlidingWindowAttn(
        seqlen_q=sq,
        seqlen_kv=skv,
        block_size_q=bq,
        block_size_kv=bkv,
        head_dim=hd,
        num_q_head=hq,
        num_kv_head=hkv,
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
        dtype=param_dtype,
        device=param_device,
    )
    
    # apply the forward pass to get student's output tensors
    on_swa(
        q, k, v,
        global_o=global_o,
        global_lse=global_lse,
        block_idx_q=bqi, 
        block_idx_kv=bkvj,
    )
    
    # check if the output tensors are correct
    assert_close(global_o, global_o_ref, atol=atol, rtol=rtol)
    assert_close(global_lse, global_lse_ref, atol=atol, rtol=rtol)
    

if __name__ == "__main__":
    pytest.main()
    