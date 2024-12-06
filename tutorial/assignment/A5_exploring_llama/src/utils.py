from typing import Optional, Sequence, List, Dict, Any, Type, Union
import os
import json
import re
import shutil

from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import wraps

import torch
import torch.nn as nn
import torch.nn.functional as F

from safetensors.torch import load_file, save_file, save_model
from huggingface_hub import split_torch_state_dict_into_shards


def load_json(path: Union[str, List[str]]) -> Union[dict, List[dict]]:
    paths = convert_to_list(path)
    
    data = None
    for path in paths:
        check_valid_path(path, ext="json")
        with open(path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            if isinstance(json_data, dict):
                if data is None:
                    data = json_data
                else:
                    assert isinstance(data, dict), f"Each previous json file contains a list of json dicts, while {path} contains only a json dict"
                    data.update(json_data)
            elif isinstance(json_data, list):
                if data is None:
                    data = json_data
                else:
                    assert isinstance(data, list), f"Each previous json file contains a json dict, while {path} contains only a list of json dicts"
                    data.extend(json_data)
            else:
                raise ValueError(f"{path} is not a valid json file")
            
    return data


def save_json(data: Union[Dict[dict, List[dict]]], path: str) -> None:
    if not path.endswith(".json"):
        raise ValueError(f"{path} is not a json file")
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


def load_jsonl(path: Union[str, List[str]]) -> List[dict]:
    paths = convert_to_list(path)
    
    data = []
    for path in paths:
        check_valid_path(path, ext="jsonl")
        with open(path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            data.extend(json.loads(line) for line in lines)
    
    return data


def save_jsonl(data: List[dict], path: str) -> None:
    if not path.endswith(".jsonl"):
        raise ValueError(f"{path} is not a jsonl file")
    if not os.path.exists(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
        
    with open(path, "w", encoding="utf-8") as f:
        for line in data:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def load_safetensors(path: Union[str, List[str]]) -> Dict[str, torch.Tensor]:
    paths = convert_to_list(path)
    
    state_dict = {}
    for path in paths:
        check_valid_path(path, ext="safetensors")
        state_dict.update(load_file(path))
    
    return state_dict


def save_safetensors(
    state_dict: Dict[str, torch.Tensor],
    save_dir: str,
    max_shard_size: Optional[int] = None, # NOTE: in unit: MB
) -> None:
    check_valid_path(
        save_dir,
        is_dir=True,
        should_exist=False,
        create_if_not_exist=True,
        empty_if_exist=True,
    )
    
    state_dict_split = split_torch_state_dict_into_shards(
        state_dict,
        max_shard_size=max_shard_size * 1024**2, # to MB
    )
    
    shard_keys = []
    for filename, tensors in state_dict_split.filename_to_tensors.items():
        # DE-BUG: save_file cannot handle shared tensors
        # shard = {tensor: state_dict[tensor].contiguous() for tensor in tensors}
        # save_file(
        #     shard,
        #     os.path.join(save_dir, filename),
        #     metadata={"format": "pt"},
        # )
        
        shard = {tensor: state_dict[tensor] for tensor in tensors}
        tmp_shard_model = nn.Module()
        tmp_shard_model.state_dict = lambda: shard
        save_model(
            tmp_shard_model,
            os.path.join(save_dir, filename),
            force_contiguous=True,
        )
        shard_keys.extend(shard.keys())
        
    if state_dict_split.is_sharded:
        t2f = state_dict_split.tensor_to_filename
        t2f_dedup = {t: f for t, f in t2f.items() if t in shard_keys}
        index = {
            # "metadata": state_dict_split.metadata, # DE-BUG: since tensors are shared, metadata is wrong, thus omitted
            "weight_map": t2f_dedup, # DE-BUG: deduplicate the shared tensors
        }
        with open(os.path.join(save_dir, "model.safetensors.index.json"), "w") as f:
            f.write(json.dumps(index, ensure_ascii=False, indent=4))


def check_valid_path(
    path: str,
    ext: Optional[str] = None,
    should_exist: bool = True,
    is_dir: bool = False,
    create_if_not_exist: bool = False,
    empty_if_exist: bool = False,
) -> None:
    if should_exist and not os.path.exists(path):
        raise ValueError(f"{path} does not exist")
        
    if is_dir:
        if os.path.exists(path) and not os.path.isdir(path):
            raise ValueError(f"{path} is not a directory")
        
        if create_if_not_exist and not os.path.exists(path):
            os.makedirs(path, exist_ok=True)
        
        if empty_if_exist and os.path.exists(path):
            shutil.rmtree(path)
            os.makedirs(path, exist_ok=True)
    else:
        if os.path.exists(path) and not os.path.isfile(path):
            raise ValueError(f"{path} is not a file")
        
        if ext is not None and not path.endswith(f".{ext}"):
            raise ValueError(f"{path} is not a {ext} file")
        
        if create_if_not_exist and not os.path.exists(path):
            os.makedirs(os.path.dirname(path))
            with open(path, "w") as _: pass
        
        if empty_if_exist and os.path.exists(path):
            os.remove(path)


def convert_to_list(x: Union[Any, Sequence[Any]]) -> List[Any]:
    if isinstance(x, (list, tuple)):
        return list(x)
    else:
        return [x]


def seconds_to_hms_str(seconds: float) -> str:
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = int(seconds % 60)
    return f"{h}h {m}m {s}s"


def find_format_keys(format_string: str) -> List[str]:
    # find all the keys in the format string: "....{key1}...{key2}..."
    pattern = r'\{([^{}]+)\}'
    return re.findall(pattern, format_string)


def safe_clone(x: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if x is None:
        return None
    else:
        return x.clone()


def format_rich_text(
    text: str,
    color: Union[str, int] = 'black',
    bright: bool = True,
    bold: bool = True,
) -> str:
    
    if isinstance(color, int):
        color = f"{color:03}" # 256 colors map
    
    if str.isnumeric(color):
        format_key = f"color({color})"
    else:
        format_key = color
        if bright:
            format_key = f"bright_{format_key}"
        if bold:
            format_key = f"bold {format_key}"
        
    return f"[{format_key}]{text}[/{format_key}]"


def multithreaded(max_workers=5):
    """Multithread Decorator
    
    NOTE: this decorator assumes that: 
        1. the iterable arguments are ONLY in the *args, thus **kwargs are always the non-iterable shared ones
        2. there's NO mutable argument that requires to be modified in-place, i.e. all of them are read-only
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            iterable_args = []
            non_iterable_args = []
            
            for arg in args:
                if isinstance(arg, (list, tuple, set)):
                    
                    iterable_args.append(arg)
                else:
                    non_iterable_args.append(arg)
            
            iterable_args = zip(*iterable_args)
            
            results = []
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                if iterable_args:
                    future_to_item = [
                        executor.submit(func, *(list(items) + non_iterable_args), **kwargs)
                        for items in iterable_args
                    ]
                    
                    for i, future in enumerate(as_completed(future_to_item)):
                        try:
                            result = future.result()
                        except Exception as exc:
                            print(f'The {i}-th result generated an exception: {exc}')
                        else:
                            results.append(result)
                else:
                    results.append(func(*args, **kwargs))
            
            return results
        return wrapper
    return decorator


def construct_offline_attn_args(
    b: int,
    sq: int,
    skv: int,
    hq: int,
    hkv: int,
    hd: int,
    qkv_pack_format: str = "q_k_v_packed",
    qkv_layout: str = "bshd",
    seqlens_q: Optional[List[int]] = None,
    seqlens_kv: Optional[List[int]] = None,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    seed: int = 42,
) -> Sequence[Optional[torch.Tensor]]:
    torch.manual_seed(seed)
    q = torch.randn((b, sq, hq, hd), dtype=dtype, device=device)
    k = torch.randn((b, skv, hkv, hd), dtype=dtype, device=device)
    v = torch.randn((b, skv, hkv, hd), dtype=dtype, device=device)
    
    if qkv_layout == "thd":
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
        
        if qkv_layout == "sbhd":
            q, k, v = [
                x.transpose(0, 1).contiguous() 
                for x in (q, k, v)
            ]
    
    if qkv_pack_format == "qkv_packed":
        assert sq == skv, "QKV pack format requires sq == skv"
        q = torch.concat((q, k, v), dim=-2)
        k, v = None, None
    elif qkv_pack_format == "q_kv_packed":
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
    bkvi: int,
    dtype: torch.dtype = torch.bfloat16,
    device: str = "cuda",
    seed: int = 42,
) -> Sequence[torch.Tensor]:
    nbq = (sq + bq - 1) // bq
    nbk = (skv + bkv - 1) // bkv
    assert bqi < nbq, f"bqi({bqi}) >= nbq({nbq})"
    assert bkvi < nbk, f"bkvi({bkvi}) >= nbk({nbk})"
    
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
    k = k[:, bkvi*bkv:(bkvi+1)*bkv, :, :]
    v = v[:, bkvi*bkv:(bkvi+1)*bkv, :, :]
    
    return q, k, v, global_o, global_lse


def construct_kvcache_args(
    b: int,
    nh: int,
    hd: int,
    qkv_layout: str,
    ops: List[Dict[str, Any]],
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
    seed: int = 42,
) -> List[Sequence[Optional[torch.Tensor]]]:
    input_tensors = []
    
    for i, op in enumerate(ops):
        if op['op'] in ("set", "append"):
            s, seqlens = op['s'], op['seqlens']
            
            torch.manual_seed(seed + i)
            k = torch.randn(b, s, nh, hd, dtype=dtype, device=device)
            v = torch.randn_like(k)
            cu_seqlens = None
            
            if qkv_layout == "bshd":
                pass
            elif qkv_layout == "sbhd":
                k, v = [x.transpose(0, 1) for x in (k, v)]
            elif qkv_layout == "thd":
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
    kv_cache_class: Type["TransformerKVCache"], # type: ignore
    config: "TransformerConfig", # type: ignore
    b: int,
    s: int,
    seqlens: Optional[List[int]] = None,
    past_seqlen_kv: int = 0,
    past_seqlens: Optional[List[int]] = None,
    dtype: torch.dtype = torch.float32,
    device: str = "cpu",
) -> Sequence[Optional[torch.Tensor]]:
    torch.manual_seed(config.init_base_seed)
    input = torch.randn(b, s, config.hidden_size, dtype=dtype, device=device)
    input_ids = torch.randint(0, config.vocab_size, (b, s), dtype=torch.int32, device=device)
    
    if seqlens is not None:
        assert config.qkv_layout.value == "thd", "if using varlen attn, the qkv_layout must be THD"
        assert b == 1, "b should be equal to 1 if using varlen attn"
        
        cu_seqlens = torch.concat([
                torch.zeros(1, dtype=torch.int32, device=device),
                torch.tensor(seqlens, dtype=torch.int32, device=device).cumsum(dim=0)
        ], dim=0).to(torch.int32)
        assert cu_seqlens[-1] == (t:=b*s), f"The sum of seqlens ({cu_seqlens[-1]}) != b*s ({t})"
    else:
        cu_seqlens = None
    
    if past_seqlen_kv > 0:
        if config.qkv_layout.value == "thd":
            assert past_seqlens is not None, "past_seqlens must be given when qkv_layout is THD and past_seqlen_kv > 0"
        kv_cache = kv_cache_class(
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
            
            if config.qkv_layout.value == "bshd":
                pass
            elif config.qkv_layout.value == "sbhd":
                past_k, past_v = [x.transpose(0, 1) for x in (past_k, past_v)]
            elif config.qkv_layout.value == "thd":
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
