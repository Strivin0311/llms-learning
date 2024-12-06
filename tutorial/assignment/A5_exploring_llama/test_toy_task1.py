# %% [markdown]
# ### Test Toy for Task 1

# %%
TEST_WITH_REF = False # NOTE: toggle this flag to `True` to enable testing with running the cells with ref
# TEST_WITH_REF = True

# %%
device = "cpu" # NOTE: you had better use "cuda", otherwise it might be very slow
# device = "cuda"

# %%
model_dir = "./model/llama_3.2_1b_instruct/"
num_shards = 1

# %% [markdown]
# #### Step0. set up the environment

# %%
import os
import json

import torch
import torch.nn.functional as F
from torch.testing import assert_close

from transformers import LlamaForCausalLM, AutoTokenizer
from transformers import logging
logging.set_verbosity_error()

# %%
if TEST_WITH_REF:
    from ref.modeling.models import (
        LlamaConfig as LlamaConfigRef,
        LlamaModel as LlamaModelRef,
    )

# %%
from src.modeling.models import (
    LlamaConfig,
    LlamaModel,
)

# %% [markdown]
# #### Step1. test LlamaConfig loading

# %%
config_file = os.path.join(model_dir, "config.json")
params_files = os.path.join(model_dir, "model.safetensors")
config_file, params_files

# %%
with open(config_file, "r") as f:
    config = json.load(f)
print(config)

# %%
llama_config_ref = None
if TEST_WITH_REF:
    llama_config_ref: LlamaConfigRef = LlamaModelRef.load_config(
        config_file, 
        param_device=device,
    )
print(llama_config_ref)

# %%
llama_config: LlamaConfig = LlamaModel.load_config(
    config_file, 
    param_device=device,
)
print(llama_config)

# %% [markdown]
# #### Step2. test LlamaModel loading

# %%
llama_tokenizer = AutoTokenizer.from_pretrained(model_dir)
llama_tokenizer.pad_token_id = llama_tokenizer.eos_token_id

llama_hf = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=llama_config.param_dtype).to(device)
print(llama_hf)

for name, param in llama_hf.named_parameters():
    print(name, param.shape, param.dtype)

# %%
if TEST_WITH_REF:
    llama_model_ref = LlamaModelRef(llama_config_ref)
    llama_model_ref.load_parameters(params_files)
    print(llama_model_ref)

    for name, param in llama_model_ref.named_parameters():
        print(name, param.shape, param.dtype)

# %%
llama_model = LlamaModel(llama_config)
llama_model.load_parameters(params_files)
print(llama_model)

for name, param in llama_model.named_parameters():
    print(name, param.shape, param.dtype)

# %% [markdown]
# #### Step3. test LlamaModel statistics APIs

# %%
punit, munit = "B", "GB"
pmap = {
    "B": 1000**3,
    "M": 1000**2,
    "K": 1000,
    "1": 1,
}
mmap = {
    "GB": 1024**3,
    "MB": 1024**2,
    "KB": 1024,
    "1": 1,
}

# %%
print(f"Total parameters: {sum(p.numel() for p in llama_hf.parameters()) / pmap[punit]:.2f} {punit}")
print(f"Memory footprint: {llama_hf.get_memory_footprint() / mmap[munit]:.2f} {munit}")

# %%
if TEST_WITH_REF:
    total_params_b, memory_gb = llama_model_ref.num_parameters(unit=punit), llama_model_ref.num_memory_footprint(unit=munit)
    print(f"Total parameters: {total_params_b:.2f} {punit}")
    print(f"Memory footprint: {memory_gb:.2f} {munit}")

# %%
total_params_b, memory_gb = llama_model.num_parameters(unit=punit), llama_model.num_memory_footprint(unit=munit)
print(f"Total parameters: {total_params_b:.2f} {punit}")
print(f"Memory footprint: {memory_gb:.2f} {munit}")

# %% [markdown]
# #### Step4. test LlamaModel forward in evaluation mode

# %%
query = "The key to life is"
input_ids = llama_tokenizer(query, return_tensors="pt").input_ids.to(device)
print(input_ids.shape, input_ids)

# %%
llama_hf.eval()
with torch.no_grad():
    outpu_hf = llama_hf.model(input_ids, return_dict=False)[0]
    logits_hf = llama_hf.lm_head(outpu_hf)
logits_hf = logits_hf[:, -1, :]
probs_hf = F.softmax(logits_hf, dim=-1)

print(probs_hf.shape, probs_hf.dtype, probs_hf)

# %%
if TEST_WITH_REF:
    llama_model_ref.eval()
    llama_model_ref.reset_kv_cache()

    with torch.no_grad():
        probs_ref = llama_model_ref(input_ids)

    print(probs_ref.shape, probs_ref.dtype, probs_ref)

# %%
llama_model.eval()
llama_model.reset_kv_cache()

with torch.no_grad():
    probs = llama_model(input_ids)

print(probs.shape, probs.dtype, probs)

# %%
try: assert_close(probs, probs_hf)
except Exception as e: print(e)

# %%
if TEST_WITH_REF:
    try: assert_close(probs, probs_ref)
    except Exception as e: print(e)

# %% [markdown]
# #### Step5. test LlamaModel forward in training mode

# %%
query = "The key to life is to be happy"
input_ids = llama_tokenizer(query, return_tensors="pt").input_ids.to(device)
labels = input_ids.clone()
print(input_ids.shape, input_ids, labels.shape, labels)

# %%
llama_hf.train()
loss_hf = llama_hf(input_ids, labels=labels).loss
print(loss_hf)

# %%
loss_ref = None
if TEST_WITH_REF:
    llama_model_ref.train()

    with torch.enable_grad():
        loss_ref = llama_model_ref(input_ids, labels=labels)

print(loss_ref)

# %%
llama_model.train()

with torch.enable_grad():
    loss = llama_model(input_ids, labels=labels)

print(loss)

# %%



