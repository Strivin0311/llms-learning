# %% [markdown]
# ### Test Toy for Task 2

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
from transformers import pipeline
from transformers import logging
logging.set_verbosity_error()

# %%
if TEST_WITH_REF:
    from ref.modeling import (
        BatchLayout as BatchLayoutRef,
        PaddingSide as PaddingSideRef,
        TruncateSide as TruncateSideRef,
        
        PromptType as PromptTypeRef,
        PromptTemplate as PromptTemplateRef,
    )
    from ref.modeling.models import (
        LlamaConfig as LlamaConfigRef,
        LlamaTokenizer as LlamaTokenizerRef,
        LlamaModel as LlamaModelRef,
    )
    from ref.inference import (
        DecodeStrategy as DecodeStrategyRef,
        InferenceConfig as InferenceConfigRef,
        InferenceAgent as InferenceAgentRef,
    )

# %%
from src.modeling import (
    BatchLayout,
    PaddingSide,
    TruncateSide,
    
    PromptType,
    PromptTemplate,
)
from src.modeling.models import (
    LlamaConfig,
    LlamaTokenizer,
    LlamaModel,
)
from src.inference import (
    DecodeStrategy,
    InferenceConfig,
    InferenceAgent,
)

# %% [markdown]
# #### Step1. load the pretrained model

# %%
config_file = os.path.join(model_dir, "config.json")
params_files = os.path.join(model_dir, "model.safetensors")

with open(config_file, "r") as f:
    config = json.load(f)

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

# %%
llama_hf = LlamaForCausalLM.from_pretrained(model_dir, torch_dtype=llama_config.param_dtype).to(device)
print(llama_hf)

# %%
llama_model_ref = None
if TEST_WITH_REF:
    llama_model_ref = LlamaModelRef(llama_config_ref)
    llama_model_ref.load_parameters(params_files)
print(llama_model_ref)

# %%
llama_model = LlamaModel(llama_config)
llama_model.load_parameters(params_files)
print(llama_model)

# %% [markdown]
# #### Step2. load the pretrained tokenizer

# %%
tokenizer_file = os.path.join(model_dir, "tokenizer.json")
tokenizer_config_file = os.path.join(model_dir, "tokenizer_config.json")

# %%
query = "The key to life is"
response1 = " not to be afraid to take risks and try new"
response2 = " not to be happy, but to be content"
prompt1 = query + response1
prompt2 = query + response2
print(prompt1, prompt2)

# %%
llama_hf_tokenizer = AutoTokenizer.from_pretrained(model_dir)
llama_hf_tokenizer.pad_token_id = llama_hf_tokenizer.eos_token_id # NOTE: it pads right by default
print(llama_hf_tokenizer.pad_token, llama_hf_tokenizer.pad_token_id, llama_hf_tokenizer.eos_token, llama_hf_tokenizer.eos_token_id, llama_hf_tokenizer.bos_token, llama_hf_tokenizer.bos_token_id)

encoded_ids_hf = llama_hf_tokenizer([prompt1, prompt2]).input_ids
print(encoded_ids_hf)

decoded_prompts = llama_hf_tokenizer.batch_decode(encoded_ids_hf, skip_special_tokens=True)
print(decoded_prompts)

# %%
if TEST_WITH_REF:
    llama_tokenizer_ref = LlamaTokenizerRef(
        vocab_file=tokenizer_file,
        config_file=tokenizer_config_file,
    )
    print(llama_tokenizer_ref)

    encoded_ids_ref = llama_tokenizer_ref.encode([prompt1, prompt2])
    print(encoded_ids_ref)
    
    decoded_prompts_ref = llama_tokenizer_ref.decode(encoded_ids_ref)
    print(decoded_prompts_ref)

# %%
llama_tokenizer = LlamaTokenizer(
    vocab_file=tokenizer_file,
    config_file=tokenizer_config_file,
)
print(llama_tokenizer)

encoded_ids = llama_tokenizer.encode([prompt1, prompt2])
print(encoded_ids)

decoded_prompts = llama_tokenizer.decode(encoded_ids)
print(decoded_prompts)

# %% [markdown]
# #### Step3. test PromptTemplate

# %%
if TEST_WITH_REF:
    prompt_template_ref = PromptTemplateRef(
        template_str="I am a {profession} named {name}, with the age of {age}.",
    )

    prompt_template_ref.set_default(name="John", age="24")
    print(prompt_template_ref.keys())
    prompt_str_ref = prompt_template_ref(profession="programmer")
    print(prompt_str_ref)

# %%
prompt_template = PromptTemplate(
    template_str="I am a {profession} named {name}, with the age of {age}.",
)

prompt_template.set_default(name="John", age="24")
print(prompt_template.keys())
prompt_str = prompt_template(profession="programmer")
print(prompt_str)

# %% [markdown]
# #### Step4. load the InferenceConfig

# %%
generation_config_file = os.path.join(model_dir, "generation_config.json")
max_new_tokens = 20
sampling_seed = 42

# %%
if TEST_WITH_REF:
    inf_config_loadded_ref = InferenceAgentRef.load_generation_config(
        generation_config_file, 
        max_new_tokens=max_new_tokens, 
        sampling_seed=sampling_seed,
        device=device,
    )
    print(inf_config_loadded_ref)

# %%
inf_config_loadded = InferenceAgent.load_generation_config(
    generation_config_file, 
    max_new_tokens=max_new_tokens, 
    sampling_seed=sampling_seed,
    device=device,
)
print(inf_config_loadded)

# %%
if TEST_WITH_REF:
    inf_config_ref = InferenceConfigRef(
        decode_strategy=DecodeStrategyRef.GREEDY,
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=0.9,
        top_k=50,
        sampling_seed=sampling_seed,
        padding_side=PaddingSideRef.LEFT,
        pad_to_multiple_of=1,
        truncate_length=None,
        truncate_side=TruncateSideRef.RIGHT,
        device=device,
    )
    print(inf_config_ref)

# %%
inf_config = InferenceConfig(
    decode_strategy=DecodeStrategy.GREEDY,
    max_new_tokens=max_new_tokens,
    temperature=1.0,
    top_p=0.9,
    top_k=50,
    sampling_seed=sampling_seed,
    padding_side=PaddingSide.LEFT,
    pad_to_multiple_of=1,
    truncate_length=None,
    truncate_side=TruncateSide.RIGHT,
    device=device,
)
print(inf_config)

# %% [markdown]
# #### Step5. load the InferenceAgent

# %%
pipe_hf = pipeline(
    "text-generation",
    model=llama_hf,
    tokenizer=llama_hf_tokenizer,
    device=device,
    do_sample=inf_config.decode_strategy == DecodeStrategy.SAMPLING,
    max_new_tokens=inf_config.max_new_tokens,
    temperature=inf_config.temperature,
    top_p=inf_config.top_p,
    top_k=inf_config.top_k,
)
print(pipe_hf)

# %%
if TEST_WITH_REF:
    inf_agent_ref = InferenceAgentRef(
        config=inf_config_ref,
        # config=inf_config_loadded_ref,
        model=llama_model_ref,
        tokenizer=llama_tokenizer_ref,
    )
    print(inf_agent_ref)

# %%
inf_agent = InferenceAgent(
    config=inf_config,
    # config=inf_config_loadded,
    model=llama_model,
    tokenizer=llama_tokenizer,
)
print(inf_agent)

# %% [markdown]
# #### Step6. test InferenceAgent on text generation

# %%
system_prompt_template = PromptTemplate(
    template_str="You're a helpful assitant on {subject}.\n",
)
context_prompt_template = PromptTemplate(
    template_str="Fill the sentence below for you to make it {adjective}.\n",
)

subject = "life"
adjective = "reasonable"

system_prompt_str = system_prompt_template(subject=subject)
context_prompt_str = context_prompt_template(adjective=adjective)
print(system_prompt_str, context_prompt_str)

# %%
querys = [
    "The key to life is",
    "The only thing we have to fear is",
    "The cat jumped on the keyboard and accidentally",
]

prompts = [
    system_prompt_str + context_prompt_str + q
    for q in querys
]
print(prompts)

# %%
prompt_dicts = pipe_hf(prompts)
for i, prompt_dict in enumerate(prompt_dicts):
    print(f"\n{'='*25} The {i}-th sample in the batch {'='*25}")
    generated_text = prompt_dict[0]["generated_text"]
    print(f"[generated_text]: {generated_text}")

# %%
if TEST_WITH_REF:
    inf_agent_ref.set_prompt(
        prompt_template=system_prompt_template,
        prompt_type=PromptTypeRef.SYSTEM,
    )
    inf_agent_ref.set_prompt(
        prompt_template=context_prompt_template,
        prompt_type=PromptTypeRef.CONTEXT,
    )
    
    prompt_dicts = inf_agent_ref(querys, subject=subject, adjective=adjective)
    for i, prompt_dict in enumerate(prompt_dicts):
        print(f"\n{'='*25} The {i}-th sample in the batch {'='*25}")
        for prompt_type, promp in prompt_dict.items():
            print(f"\n[{prompt_type}]: {promp}")

# %%
inf_agent.set_prompt(
    prompt_template=system_prompt_template,
    prompt_type=PromptType.SYSTEM,
)
inf_agent.set_prompt(
    prompt_template=context_prompt_template,
    prompt_type=PromptType.CONTEXT,
)
prompt_dicts = inf_agent(querys, subject=subject, adjective=adjective)
for i, prompt_dict in enumerate(prompt_dicts):
    print(f"\n{'='*25} The {i}-th sample in the batch {'='*25}")
    for prompt_type, promp in prompt_dict.items():
        print(f"\n[{prompt_type}]: {promp}")

# %%



