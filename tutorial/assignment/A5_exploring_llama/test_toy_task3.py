# %% [markdown]
# ### Test Toy for Task 3

# %%
TEST_WITH_REF = False # NOTE: toggle this flag to `True` to enable testing with running the cells with ref
# TEST_WITH_REF = True

# %%
# device = "cpu" # NOTE: you had better use "cuda", otherwise it might be very slow
device = "cuda:0"
device_ref = "cuda:1" # NOTE: you had better put ref training codes into another GPU if available, otherwise it might easily run out of memory

# %%
model_dir = "./model/llama_3.2_1b_instruct/"
num_shards = 1

# %% [markdown]
# #### Step0. set up the environment

# %%
import os
import json
from datetime import datetime

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
from torch.testing import assert_close

from transformers import LlamaForCausalLM, AutoTokenizer
from transformers import logging
logging.set_verbosity_error()

# %%
if TEST_WITH_REF:
    from ref.modeling import (
        BatchLayout as BatchLayoutRef,
        PaddingSide as PaddingSideRef,
        TruncateSide as TruncateSideRef,
        
        PromptTemplate as PromptTemplateRef,
    )
    from ref.modeling.models import (
        LlamaConfig as LlamaConfigRef,
        LlamaTokenizer as LlamaTokenizerRef,
        LlamaModel as LlamaModelRef,
    )
    from ref.modeling.datasets import (
        BaseDatasetConfig as BaseDatasetConfigRef,
        QADatasetConfig as QADatasetConfigRef,
        QADataset as QADatasetRef,
        ChatDatasetConfig as ChatDatasetConfigRef,
        ChatDataset as ChatDatasetRef,
    )
    from ref.training import (
        OptimizerType as OptimizerTypeRef,
        TrainLogType as TrainLogTypeRef,
        BaseTrainConfig as BaseTrainConfigRef,
        BaseTrainer as BaseTrainerRef,
        LoRATrainConfig as LoRATrainConfigRef,
        LoRATrainer as LoRATrainerRef,
    )

# %%
from src.modeling import (
    BatchLayout,
    PaddingSide,
    TruncateSide,
    
    PromptTemplate,
)
from src.modeling.models import (
    BaseTokenizer,
    BaseModel,
    LlamaConfig,
    LlamaTokenizer,
    LlamaModel,
)
from src.modeling.datasets import (
    BaseDatasetConfig,
    QADatasetConfig,
    QADataset,
    ChatDatasetConfig,
    ChatDataset,
)
from src.training import (
    OptimizerType,
    TrainLogType,
    BaseTrainConfig,
    BaseTrainer,
    LoRATrainConfig,
    LoRATrainer,
)

# %% [markdown]
# #### Step1. load the pretrained tokenizer

# %%
tokenizer_file = os.path.join(model_dir, "tokenizer.json")
tokenizer_config_file = os.path.join(model_dir, "tokenizer_config.json")

# %%
if TEST_WITH_REF:
    llama_tokenizer_ref = LlamaTokenizerRef(
        vocab_file=tokenizer_file,
        config_file=tokenizer_config_file,
    )
    print(llama_tokenizer_ref)

# %%
llama_tokenizer = LlamaTokenizer(
    vocab_file=tokenizer_file,
    config_file=tokenizer_config_file,
)
print(llama_tokenizer)

# %% [markdown]
# #### Step2. build the QADataset

# %%
qa_train_data_file = "./data/qa/qa_train.jsonl"
qa_eval_data_file = "./data/qa/qa_eval.jsonl"
qa_test_data_file = "./data/qa/qa_test.jsonl"

# %%
if TEST_WITH_REF:
    qa_dataset_config_ref = QADatasetConfigRef(
        batch_size=2,
        seq_len=512,
        drop_last_incomplete_batch=True,
        padding_side=PaddingSideRef.LEFT,
        truncate_side=TruncateSideRef.RIGHT,
        device=device_ref,
    )
    print(qa_dataset_config_ref)

# %%
qa_dataset_config = QADatasetConfig(
    batch_size=2,
    seq_len=512,
    drop_last_incomplete_batch=True,
    padding_side=PaddingSide.LEFT,
    truncate_side=TruncateSide.RIGHT,
    device=device,
)
print(qa_dataset_config)

# %%
if TEST_WITH_REF:
    qa_train_dataset_ref = QADatasetRef(
        config=qa_dataset_config_ref,
        tokenizer=llama_tokenizer_ref,
        data_files=qa_train_data_file,
    )
    qa_eval_dataset_ref = QADatasetRef(
        config=qa_dataset_config_ref,
        tokenizer=llama_tokenizer_ref,
        data_files=qa_eval_data_file,
    )
    qa_test_dataset_ref = QADatasetRef(
        config=qa_dataset_config_ref,
        tokenizer=llama_tokenizer_ref,
        data_files=qa_test_data_file,
    )

    print(qa_train_dataset_ref.num_samples(), qa_train_dataset_ref.num_batchs(), qa_eval_dataset_ref.num_samples(), qa_eval_dataset_ref.num_batchs(), qa_test_dataset_ref.num_samples(), qa_test_dataset_ref.num_batchs())

# %%
qa_train_dataset = QADataset(
    config=qa_dataset_config,
    tokenizer=llama_tokenizer,
    data_files=qa_train_data_file,
)
qa_eval_dataset = QADataset(
    config=qa_dataset_config,
    tokenizer=llama_tokenizer,
    data_files=qa_eval_data_file,
)
qa_test_dataset = QADataset(
    config=qa_dataset_config,
    tokenizer=llama_tokenizer,
    data_files=qa_test_data_file,
)

print(qa_train_dataset.num_samples(), qa_train_dataset.num_batchs(), qa_eval_dataset.num_samples(), qa_eval_dataset.num_batchs(), qa_test_dataset.num_samples(), qa_test_dataset.num_batchs())

# %%
if TEST_WITH_REF:
    sample_ref = qa_test_dataset_ref.sample(200)
    print(sample_ref)
    
    batch_ref = qa_test_dataset_ref.batch(100)
    
    print(batch_ref[qa_dataset_config_ref.samples_key])

    print(batch_ref[qa_dataset_config_ref.input_ids_key], batch_ref[qa_dataset_config_ref.input_ids_key].shape)

    print(batch_ref[qa_dataset_config_ref.labels_key], batch_ref[qa_dataset_config_ref.labels_key].shape)

    print(llama_tokenizer_ref.decode(batch_ref[qa_dataset_config_ref.input_ids_key][0]))

    labels_ref = batch_ref[qa_dataset_config_ref.labels_key][0].clone()
    labels_ref[labels_ref == qa_dataset_config_ref.ignore_idx] = llama_tokenizer_ref.bos_id \
        if qa_dataset_config_ref.padding_side == PaddingSideRef.LEFT else llama_tokenizer_ref.eos_id
    print(llama_tokenizer_ref.decode(labels_ref))

# %%
sample = qa_test_dataset.sample(200)
print(sample)

batch = qa_test_dataset.batch(100)

print(batch[qa_dataset_config.samples_key])

print(batch[qa_dataset_config.input_ids_key], batch[qa_dataset_config.input_ids_key].shape)

print(batch[qa_dataset_config.labels_key], batch[qa_dataset_config.labels_key].shape)

print(llama_tokenizer.decode(batch[qa_dataset_config.input_ids_key][0]))

labels = batch[qa_dataset_config.labels_key][0].clone()
labels[labels == qa_dataset_config.ignore_idx] = llama_tokenizer.bos_id \
    if qa_dataset_config.padding_side == PaddingSide.LEFT else llama_tokenizer.eos_id
print(llama_tokenizer.decode(labels))

# %% [markdown]
# #### Step3. build the ChatDataset

# %%
chat_test_data_file = "./data/chat/chat_test.jsonl"
chat_train_data_file = "./data/chat/chat_train.jsonl"
chat_eval_data_file = "./data/chat/chat_eval.jsonl"

# %%
if TEST_WITH_REF:
    chat_dataset_config_ref = ChatDatasetConfigRef(
        batch_size=2,
        seq_len=1024,
        drop_last_incomplete_batch=True,
        padding_side=PaddingSideRef.LEFT,
        truncate_side=TruncateSideRef.RIGHT,
        device=device_ref,
    )
    print(chat_dataset_config_ref)

# %%
chat_dataset_config = ChatDatasetConfig(
    batch_size=2,
    seq_len=1024,
    drop_last_incomplete_batch=True,
    padding_side=PaddingSide.LEFT,
    truncate_side=TruncateSide.RIGHT,
    device=device,
)
print(chat_dataset_config)

# %%
if TEST_WITH_REF:
    chat_train_dataset_ref = ChatDatasetRef(
        config=chat_dataset_config_ref,
        tokenizer=llama_tokenizer_ref,
        data_files=chat_train_data_file,
    )
    chat_eval_dataset_ref = ChatDatasetRef(
        config=chat_dataset_config_ref,
        tokenizer=llama_tokenizer_ref,
        data_files=chat_eval_data_file,
    )
    chat_test_dataset_ref = ChatDatasetRef(
        config=chat_dataset_config_ref,
        tokenizer=llama_tokenizer_ref,
        data_files=chat_test_data_file,
    )

    print(chat_train_dataset_ref.num_samples(), chat_train_dataset_ref.num_batchs(), chat_eval_dataset_ref.num_samples(), chat_eval_dataset_ref.num_batchs(), chat_test_dataset_ref.num_samples(), chat_test_dataset_ref.num_batchs())

# %%
chat_train_dataset = ChatDataset(
    config=chat_dataset_config,
    tokenizer=llama_tokenizer,
    data_files=chat_train_data_file,
)
chat_eval_dataset = ChatDataset(
    config=chat_dataset_config,
    tokenizer=llama_tokenizer,
    data_files=chat_eval_data_file,
)
chat_test_dataset = ChatDataset(
    config=chat_dataset_config,
    tokenizer=llama_tokenizer,
    data_files=chat_test_data_file,
)

print(chat_train_dataset.num_samples(), chat_train_dataset.num_batchs(), chat_eval_dataset.num_samples(), chat_eval_dataset.num_batchs(), chat_test_dataset.num_samples(), chat_test_dataset.num_batchs())

# %%
if TEST_WITH_REF:
    sample_ref = chat_test_dataset_ref.sample(200)
    print(sample_ref)
    
    batch_ref = chat_test_dataset_ref.batch(100)
    
    print(batch_ref[chat_dataset_config_ref.samples_key])

    print(batch_ref[chat_dataset_config_ref.input_ids_key], batch_ref[chat_dataset_config_ref.input_ids_key].shape)

    print(batch_ref[chat_dataset_config_ref.labels_key], batch_ref[chat_dataset_config_ref.labels_key].shape)

    print(llama_tokenizer_ref.decode(batch_ref[chat_dataset_config_ref.input_ids_key][0]))

    labels_ref = batch_ref[chat_dataset_config_ref.labels_key][0].clone()
    labels_ref[labels_ref == chat_dataset_config_ref.ignore_idx] = llama_tokenizer_ref.bos_id \
        if chat_dataset_config_ref.padding_side == PaddingSideRef.LEFT else llama_tokenizer_ref.eos_id
    print(llama_tokenizer_ref.decode(labels_ref))

# %%
sample = chat_test_dataset.sample(200)
print(sample)

batch = chat_test_dataset.batch(100)

print(batch[chat_dataset_config.samples_key])

print(batch[chat_dataset_config.input_ids_key], batch[chat_dataset_config.input_ids_key].shape)

print(batch[chat_dataset_config.labels_key], batch[chat_dataset_config.labels_key].shape)

print(llama_tokenizer.decode(batch[chat_dataset_config.input_ids_key][0]))

labels = batch[chat_dataset_config.labels_key][0].clone()
labels[labels == chat_dataset_config.ignore_idx] = llama_tokenizer.bos_id \
    if chat_dataset_config.padding_side == PaddingSide.LEFT else llama_tokenizer.eos_id
print(llama_tokenizer.decode(labels))

# %% [markdown]
# #### Step4. build the LoRATrainConfig for QA task

# %%
save_ckpt_dir_qa = "./ckpt/lora_qa/"
load_ckpt_dirs_qa = None # NOTE: from scratch

# %%
if TEST_WITH_REF:
    qa_train_config_ref = LoRATrainConfigRef(
        train_steps=10,
        eval_interval=2,
        eval_steps=1,
        
        shuffle=True,
        shuffle_seed=42,
        optimizer_type=OptimizerTypeRef.ADAMW,
        learning_rate=2e-4,
        
        load_ckpt_dirs=load_ckpt_dirs_qa,
        save_interval=4,
        save_ckpt_dir=save_ckpt_dir_qa,
        max_shard_size=1024, # MB
        save_only_lora=True,
        
        log_interval=1,
        log_types=(TrainLogTypeRef.TERMINAL,),
        # log_types=(TrainLogTypeRef.TERMINAL, TrainLogTypeRef.TENSORBOARD, TrainLogTypeRef.WANDB),
        log_kwargs={
            "wandb_api_key": "", # NOTE: replace this to your own wandb api key
            "wandb_project": "nju-llm-course", # NOTE: replace this to your own wandb project name
            "wandb_name": f"lora_qa_ref_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", # NOTE: replace this to your own wandb job name
            "wandb_dir": "./wandb/lora_qa_ref", # NOTE: replace this to your own wandb local dir
            
            "tensorboard_log_dir": "./tensorboard/lora_qa_ref", # NOTE: replace this to your own tensorboard local log dir
        },
        
        device=device_ref,
        
        lora_weight_A_pattern="lora_weight_A",
        lora_weight_B_pattern="lora_weight_B",
    )
    print(qa_train_config_ref)

# %%
qa_train_config = LoRATrainConfig(
    train_steps=10,
    eval_interval=2,
    eval_steps=1,
    
    shuffle=True,
    shuffle_seed=42,
    optimizer_type=OptimizerType.ADAMW,
    learning_rate=2e-4,
    
    load_ckpt_dirs=load_ckpt_dirs_qa,
    save_interval=4,
    save_ckpt_dir=save_ckpt_dir_qa,
    max_shard_size=1024, # MB
    save_only_lora=True,
    
    log_interval=1,
    log_types=(TrainLogType.TERMINAL,),
    # log_types=(TrainLogType.TERMINAL, TrainLogType.TENSORBOARD, TrainLogType.WANDB), # NOTE: to use multiple loggers if you've supported
    log_kwargs={
        "wandb_api_key": "", # NOTE: replace this to your own wandb api key
        "wandb_project": "nju-llm-course", # NOTE: replace this to your own wandb project name
        "wandb_name": f"lora_qa_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", # NOTE: replace this to your own wandb job name
        "wandb_dir": "./wandb/lora_qa", # NOTE: replace this to your own wandb local dir
        
        "tensorboard_log_dir": "./tensorboard/lora_qa", # NOTE: replace this to your own tensorboard local log dir
    },
    
    device=device,
    
    lora_weight_A_pattern="lora_weight_A", # NOTE: replace this to your own pattern
    lora_weight_B_pattern="lora_weight_B", # NOTE: replace this to your own pattern
)
print(qa_train_config)

# %% [markdown]
# #### Step5. build the LoRATrainer to finetune the model on QA task

# %%
punit, munit = "M", "MB"
lora_rank_qa = 8

# %%
config_file = os.path.join(model_dir, "config.json")
params_files = os.path.join(model_dir, "model.safetensors")

with open(config_file, "r") as f:
    config = json.load(f)

# %%
llama_model_qa_ref = None
if TEST_WITH_REF:
    llama_config_qa_ref: LlamaConfigRef = LlamaModelRef.load_config(
        config_file,
        lora_rank=lora_rank_qa,
        param_device=device_ref,
    )
    print(llama_config_qa_ref)
    
    llama_model_qa_ref = LlamaModelRef(llama_config_qa_ref)
    llama_model_qa_ref.load_parameters(params_files)
print(llama_model_qa_ref)

# %%
llama_config_qa: LlamaConfig = LlamaModel.load_config(
    config_file,
    lora_rank=lora_rank_qa,
    param_device=device,
)
print(llama_config_qa)

llama_model_qa = LlamaModel(llama_config_qa)
llama_model_qa.load_parameters(params_files)

# %%
if TEST_WITH_REF:
    total_params_b, memory_gb = llama_model_qa_ref.num_parameters(learnable_only=True, unit=punit), llama_model_qa_ref.num_memory_footprint(unit=munit)
    print(f"Total trainable parameters Before: {total_params_b:.2f} {punit}")
    print(f"Memory footprint Before: {memory_gb:.2f} {munit}")

    lora_qa_trainer_ref = LoRATrainerRef(
        config=qa_train_config_ref,
        model=llama_model_qa_ref,
        tokenizer=llama_tokenizer_ref,
        train_dataset=qa_train_dataset_ref,
        eval_dataset=qa_eval_dataset_ref,
    )
    print(lora_qa_trainer_ref)
    
    total_params_b, memory_gb = llama_model_qa_ref.num_parameters(learnable_only=True, unit=punit), llama_model_qa_ref.num_memory_footprint(unit=munit)
    print(f"Total trainable parameters After: {total_params_b:.2f} {punit}")
    print(f"Memory footprint After: {memory_gb:.2f} {munit}")

# %%
total_params_b, memory_gb = llama_model_qa.num_parameters(learnable_only=True, unit=punit), llama_model_qa.num_memory_footprint(unit=munit)
print(f"Total trainable parameters Before: {total_params_b:.2f} {punit}")
print(f"Memory footprint Before: {memory_gb:.2f} {munit}")

lora_qa_trainer = LoRATrainer(
    config=qa_train_config,
    model=llama_model_qa,
    tokenizer=llama_tokenizer,
    train_dataset=qa_train_dataset,
    eval_dataset=qa_eval_dataset,
)
print(lora_qa_trainer)

total_params_b, memory_gb = llama_model_qa.num_parameters(learnable_only=True, unit=punit), llama_model_qa.num_memory_footprint(unit=munit)
print(f"Total trainable parameters After: {total_params_b:.2f} {punit}")
print(f"Memory footprint After: {memory_gb:.2f} {munit}")

# %%
if TEST_WITH_REF:
    lora_qa_trainer_ref.run()

# %%
lora_qa_trainer.run()

# %% [markdown]
# #### Step6. build the LoRATrainConfig for Chatbot task

# %%
save_ckpt_dir_chat = "./ckpt/lora_chat/"
load_ckpt_dirs_chat = None # NOTE: from scratch

# %%
if TEST_WITH_REF:
    chat_train_config_ref = LoRATrainConfigRef(
        train_steps=10,
        eval_interval=2,
        eval_steps=1,
        
        shuffle=True,
        shuffle_seed=42,
        optimizer_type=OptimizerTypeRef.ADAMW,
        learning_rate=2e-4,
        
        load_ckpt_dirs=load_ckpt_dirs_chat,
        save_interval=4,
        save_ckpt_dir=save_ckpt_dir_chat,
        max_shard_size=1024, # MB
        save_only_lora=True,
        
        log_interval=1,
        log_types=(TrainLogTypeRef.TERMINAL,),
        # log_types=(TrainLogTypeRef.TERMINAL, TrainLogTypeRef.TENSORBOARD, TrainLogTypeRef.WANDB),
        log_kwargs={
            "wandb_api_key": "", # NOTE: replace this to your own wandb api key
            "wandb_project": "nju-llm-course", # NOTE: replace this to your own wandb project name
            "wandb_name": f"lora_chat_ref_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", # NOTE: replace this to your own wandb job name
            "wandb_dir": "./wandb/lora_chat_ref", # NOTE: replace this to your own wandb local dir
            
            "tensorboard_log_dir": "./tensorboard/lora_chat_ref", # NOTE: replace this to your own tensorboard local log dir
        },
        
        device=device_ref,
        
        lora_weight_A_pattern="lora_weight_A",
        lora_weight_B_pattern="lora_weight_B",
    )
    print(chat_train_config_ref)

# %%
chat_train_config = LoRATrainConfig(
    train_steps=10,
    eval_interval=2,
    eval_steps=1,
    
    shuffle=True,
    shuffle_seed=42,
    optimizer_type=OptimizerType.ADAMW,
    learning_rate=2e-4,
    
    load_ckpt_dirs=load_ckpt_dirs_chat,
    save_interval=4,
    save_ckpt_dir=save_ckpt_dir_chat,
    max_shard_size=1024, # MB
    save_only_lora=True,
    
    log_interval=1,
    log_types=(TrainLogType.TERMINAL,),
    # log_types=(TrainLogType.TERMINAL, TrainLogType.TENSORBOARD, TrainLogType.WANDB), # NOTE: to use multiple loggers if you've supported
    log_kwargs={
        "wandb_api_key": "", # NOTE: replace this to your own wandb api key
        "wandb_project": "nju-llm-course", # NOTE: replace this to your own wandb project name
        "wandb_name": f"lora_chat_{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", # NOTE: replace this to your own wandb job name
        "wandb_dir": "./wandb/lora_chat", # NOTE: replace this to your own wandb local dir
        
        "tensorboard_log_dir": "./tensorboard/lora_chat", # NOTE: replace this to your own tensorboard local log dir
    },
    
    device=device,
    
    lora_weight_A_pattern="lora_weight_A", # NOTE: replace this to your own pattern
    lora_weight_B_pattern="lora_weight_B", # NOTE: replace this to your own pattern
)
print(chat_train_config)

# %% [markdown]
# #### Step7. build the LoRATrainer to finetune the model on Chatbot task

# %%
punit, munit = "M", "MB"
lora_rank_chat = 16

# %%
config_file = os.path.join(model_dir, "config.json")
params_files = os.path.join(model_dir, "model.safetensors")

with open(config_file, "r") as f:
    config = json.load(f)

# %%
llama_model_chat_ref = None
if TEST_WITH_REF:
    llama_config_chat_ref: LlamaConfigRef = LlamaModelRef.load_config(
        config_file,
        lora_rank=lora_rank_chat,
        param_device=device_ref,
    )
    print(llama_config_chat_ref)
    
    llama_model_chat_ref = LlamaModelRef(llama_config_chat_ref)
    llama_model_chat_ref.load_parameters(params_files)
print(llama_model_chat_ref)

# %%
llama_config_chat: LlamaConfig = LlamaModel.load_config(
    config_file,
    lora_rank=lora_rank_chat,
    param_device=device,
)
print(llama_config_chat)

llama_model_chat = LlamaModel(llama_config_chat)
llama_model_chat.load_parameters(params_files)

# %%
if TEST_WITH_REF:
    total_params_b, memory_gb = llama_model_chat_ref.num_parameters(learnable_only=True, unit=punit), llama_model_chat_ref.num_memory_footprint(unit=munit)
    print(f"Total trainable parameters Before: {total_params_b:.2f} {punit}")
    print(f"Memory footprint Before: {memory_gb:.2f} {munit}")

    lora_chat_trainer_ref = LoRATrainerRef(
        config=chat_train_config_ref,
        model=llama_model_chat_ref,
        tokenizer=llama_tokenizer_ref,
        train_dataset=chat_train_dataset_ref,
        eval_dataset=chat_eval_dataset_ref,
    )
    print(lora_chat_trainer_ref)
    
    total_params_b, memory_gb = llama_model_chat_ref.num_parameters(learnable_only=True, unit=punit), llama_model_chat_ref.num_memory_footprint(unit=munit)
    print(f"Total trainable parameters After: {total_params_b:.2f} {punit}")
    print(f"Memory footprint After: {memory_gb:.2f} {munit}")

# %%
total_params_b, memory_gb = llama_model_chat.num_parameters(learnable_only=True, unit=punit), llama_model_chat.num_memory_footprint(unit=munit)
print(f"Total trainable parameters Before: {total_params_b:.2f} {punit}")
print(f"Memory footprint Before: {memory_gb:.2f} {munit}")

lora_chat_trainer = LoRATrainer(
    config=qa_train_config,
    model=llama_model_chat,
    tokenizer=llama_tokenizer,
    train_dataset=chat_train_dataset,
    eval_dataset=chat_eval_dataset,
)
print(lora_chat_trainer)

total_params_b, memory_gb = llama_model_chat.num_parameters(learnable_only=True, unit=punit), llama_model_chat.num_memory_footprint(unit=munit)
print(f"Total trainable parameters After: {total_params_b:.2f} {punit}")
print(f"Memory footprint After: {memory_gb:.2f} {munit}")

# %%
if TEST_WITH_REF:
    lora_chat_trainer_ref.run()

# %%
lora_chat_trainer.run()

# %% [markdown]
# #### Appendix. show different training logging examples by the `ref`

# %% [markdown]
# considering most of the students might have no experience in using tools like rich, wandb, tensorboard, etc, to polish / visualize the training log,
# 
# we provide some training log examples by the `ref` for you to have a quick look

# %%
asset_dir = "./asset/"

# %% [markdown]
# ##### LogType.TERMINAL

# %%
lora_qa_terminal_log_example_path = os.path.join(asset_dir, "lora_qa_terminal_log_example.png")
# Image.open(lora_qa_terminal_log_example_path)

# %%
lora_chat_terminal_log_example_path = os.path.join(asset_dir, "lora_chat_terminal_log_example.png")
# Image.open(lora_chat_terminal_log_example_path)

# %% [markdown]
# ##### LogType.TENSORBOARD

# %%
lora_qa_chat_tensorboard_log_example_path = os.path.join(asset_dir, "lora_qa+chat_tensorboard_log_example.png")
# Image.open(lora_qa_chat_tensorboard_log_example_path)

# %% [markdown]
# ##### LogType.WANDB

# %%
lora_qa_chat_wandb_log_example_path = os.path.join(asset_dir, "lora_qa+chat_wandb_log_example.png")
# Image.open(lora_qa_chat_wandb_log_example_path)

# %%



