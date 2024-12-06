### Task 3: LoRA Trainer (plus bonus🔥)


#### TODO

You are required to implement two kinds of pytorch modules, each kind containing a base parent module and a child module, respectively:

* the first kind is for (training) dataset, where the parent module is `QADataset` in `src/modeling/datasets/qa.py`, and the child module is `ChatDataset` in `src/modeling/datasets/chat.py`.
* the second kind is for trainer, where the parent module is `BaseTrainer` in `src/training/base.py`, and the child module `LoRATrainer` in `src/training/lora.py`.


#### Explanation

* The `BaseDataset` defines the necessary APIs for preparing the (training) dataset, but all are leaved as abstract methods to be implemented by you in the subclasses `QADataset` and `ChatDataset`, which are briefly depicted as below:
    * `num_samples()`: return the number of samples in the dataset
    * `sample(idx: int)`: return a sample as a dictionary with the index `idx` in the dataset
    * `samples()`: return a iterator that yields samples one by one until all samples have been iterated
    * `num_batchs()`: return the number of batchs in the dataset
    * `batch(idx: int)`: return a batch as a dictionary with the index `idx` in the dataset
    * `batches()`: return a iterator that yields batches one by one until all batches have been iterated
    * `shuffle(seed: Optional[int] = None)`: shuffle the dataset including the samples and batches with the optional random seed `seed`
* As for the `QADataset`, it is the subclass of `BaseDataset` which implements all the APIs above to prepare the (training) dataset for the Question-Answering task, one basic sequence-to-sequence NLP task for language models, **where the input is the question field and the training target is the answer field**. More specifically, it requires a `QADatasetConfig` dataclass object, a `BaseTokenizer` tokenizer, and path(s) to the file(s) with the data in `.jsonl` format, where each line represents a sample as a json dict, for example:
    ```json
    {
        "question": "Imagine yourself as a film director. Write the plot summary for a quirky romantic comedy movie set in a remote tropical island. Include details about the main characters, their personality traits, backgrounds, challenges they face, and how their relationship develops over the course of the story. Make it creative and humorous.", 
        "answer": "The film follows Jade, an eccentric free-spirited woman in her late 20s who travels to a small tropical island to escape the troubles of her everyday life. She settles into a simple hut by the beach and spends her days swimming, exploring the jungle, and partaking in the island's laidback village life.  \n\nThe male lead is Kai, a serious marine biologist in his early 30s, who lives on the island conducting research on the local coral reef ecosystem. Kai is a man of routine, discipline, and habit, focusing strictly on his work with little interest in socializing."
    }
    ```
* We've already prepared the example qa data for you in `./data/qa/` and split into three-folds as `qa_train.jsonl`, `qa_eval.jsonl` and `qa_test.jsonl` , representing the training dataset, evaluation dataset and testing dataset, respectively. And what you need to do is parse these raw samples when initializing the `QADataset` object, as well as processing them into batches of token ids and labels with `truncation` and `padding` in the similar way in [task2](./task2.md) for the `InferenceAgent` (*See the HF Transformers Data Collator in [References](#references)*). And the final output for the `batch(idx: int)` API should look like:
    ```python
    {
        "input_ids": torch.tensor([[...]]),
        "cu_seqlens": torch.tensor([...]), # or None
        "labels": torch.tensor([[...]]),
        "samples": [
            {"question": "...", "answer": "..."},
            {"question": "...", "answer": "..."},
            ...,
        ]
    }
    ```
* Similarly, `ChatDataset` is to prepare (training) dataset for Chatbot task, whose sample dict contains only a "conversations" field, which points to a list of dicts, recording a dialogue between a human with the role `user` and an inference agent with the role `chatbot`, for example:
    ```json
    {
        "conversations": [
            {"role": "user", "content": "Hi"},
            {"role": "chatbot", "content": "Hello"},
            {"role": "user", "content": "How are you?"},
            {"role": "chatbot", "content": "I'm fine, thanks. And you?"},
            {"role": "user", "content": "I'm good too. I have something to ask you."},
            {"role": "chatbot", "content": "What is it?"},
            {"role": "user", "content": "I want to know what is the meaning of life."}
            {"role": "chatbot", "content": "The meaning of life is to be happy."}
        ]
    }
    ```
* So, it is an extension to the basic qa task, **where the input covers all of the contents from the user, while the training target includes all of the contents from the chatbot in one dialogue**. To reduce redundancy, we directly choose to inherit from `QADataset` and what you need to do is design the minimal set of "interfaces" smartly for `QADataset` to leave to the `ChatDataset` to overwite.
* Here's a full table that informs you the details about each configuration in the `BaseDatasetConfig` dataclass as follows (*Note: "Required" means the argument must be provided with non-`None` values during initializationm and "Fixed" means the argument cannot be set during initialization and retain their default values.*):

    | **Config Name**                 | **Type**                      | **Default**                    | **Required** | **Fixed** | **Description**                                                                                                                                      |
    |----------------------------------|-------------------------------|--------------------------------|--------------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------|
    | `seq_len`                        | `int`                         | `None`                         | `True`       | `False`   | The sequence length of every sequence to align to in each batch.                                      |
    | `batch_size`                     | `int`                         | `None`                         | `True`       | `False`   | The batch size of each inputs.                         |
    | `batch_layout`                   | `BatchLayout`                 | `BatchLayout.STACK`            | `False`      | `True`    | Specifies how the input batch is organized. Currently, only stacking is allowed for simplicity.                                                     |
    | `padding_side`                   | `PaddingSide`                 | `PaddingSide.LEFT`             | `False`      | `False`   | Specifies the side (left or right) on which padding is added to sequences that are shorted than `seq_len`. Default is padding on the left.                                         |
    | `truncate_side`                  | `TruncateSide`                | `TruncateSide.RIGHT`           | `False`      | `False`   | Specifies which side (left or right) to truncate each too-long input sequence from if they exceed the `seq_len`.                               |
    | `drop_last_incomplete_batch`     | `bool`                        | `True`                         | `False`      | `False`   | Whether to drop the last batch if it is incomplete (e.g., has fewer than `batch_size` samples).                                                      |
    | `samples_key`                    | `str`                         | `"samples"`                    | `False`      | `False`   | The key in one batch dictionary pointing to the raw list of sample dicts that form this batch.                                                                                     |
    | `input_ids_key`                  | `str`                         | `"input_ids"`                  | `False`      | `False`   | The key in one batch dictionary pointing to the input token ids for this batch.                                                                         |
    | `labels_key`                     | `str`                         | `"labels"`                     | `False`      | `False`   | The key in one batch dictionary pointing to the target token ids for this batch.                                                                            |
    | `cu_seqlens_key`                 | `str`                         | `"cu_seqlens"`                 | `False`      | `False`   | The key in one batch dictionary pointing to the optional cumulative sequence lengths for this batch (*NOTE: since the batch_layout is fixed to `BatchLayout.STACK`, this key is always `None` for this task*).                                                   |
    | `ignore_idx`                     | `int`                         | `-100`                         | `False`      | `False`   | The index to ignore during loss computation.                                                                         |
    | `prefix_template`                | `PromptTemplate`              | `PromptTemplate("[{prefix}]: ")`| `False`      | `False`   | The template used to prefix a prompt, e.g. in qa task, the question prompt may prepend a prefix as `"[QUESTION]: "` if the template is `"[{prefix}]: "` and `prefix="QUESTION"`.                                                  |
    | `sep_str`                        | `str`                         | `"\n"`                          | `False`      | `False`   | The separator string used to separate different parts of prompts, e.g. in chatbot task, not only the user prompt and chatbot response, but also two adjacent conversation can be separated with this `sep_str` as `"[USER]: aaa{sep_str}[CHATBOT]: bbb{sep_str}[USER]: ccc{sep_str}[CHATBOT]: ddd"`.                                                      |
    | `device`                         | `str`                         | `"cpu"`                        | `False`      | `False`   | The device (`"cpu"` or `"cuda"`) where the batches of data should be placed.    |
* And as for the `QADatasetConfig` and the `ChatDatasetConfig` as the sub-dataclasses of `BaseDatasetConfig`, we might only need to additionally provide specific (fixed) fields such as the keys in the sample dict.
* With datasets prepared, the `BaseTrainer` defines the necessary APIs for LLM training, and provides the default general implementations for these APIs, which are detailedly depicted as below:
    * `__init__(config: BaseTrainConfig, model: BaseModel, tokenizer: BaseTokenizer, train_dataset: BaseDataset, eval_dataset: Optional[BaseDataset] = None)`: intialize the trainer given the training configurations, model, tokenizer, training dataset and optional evaluation dataset, roughly including:
        * 1. setting the trainable parameters, loading the checkpoint to resume training if needed, etc.
        * 2. building the cyclic data iterators to deal with the situation when the iteration steps are large than the number of batches.
        * 3. building the optimizer with the given algorithm type `optimizer_type` and the corresponding hyperparameters including `learning_rate`, `momentum`, `betas`, `weight_decay`, etc.
        * 4. initializing the logger(s) such as `wandb` (*See WanDB Tutorials in [References](#references) for more details*) and `tensorboard` (*See Tensorboard SummaryWriter in [References](#references) for more details*), with the given types `log_types` and the optional keyword arguments `log_kwargs`.
    * `run()`: run the whole training steps in a loop, until the stopping criterion is met, and remember that this is an **one-time** API, and you have to re-initialize a new trainer if you need to rerun (*If you're not familiar with the training loop, your can refer to `PyTorch Training Loop` in [References](#references) for more details*).
    * `_train_step()`: one training step pass, called in `self.run()`, including the following sub-steps:
        * 1. feed a batch of data to the model to apply forward pass with gradient tracking **enabled** to get the training loss
        * 2. apply backward pass to compute the gradients
        * 3. let the optimizer update the model parameters with the gradients
        * 4. return the training loss
    * `_eval_step()`: one evaluation step pass, called in `self.run()` when the evaluation criterion is met, which basically feeds a batch of data to the model to apply forward pass with gradient tracking **disabled** to get the evaluation loss returned
    * `_load_ckpt()`: load the model from the pretrained checkpoint directory (or directories) to resume training if needed, called in `__init__()`
    * `_save_ckpt(step: int)`: save the model at the current training step to the checkpoint directory, called in `self.run()` when the saving criterion is met
* To support lora fine-tuning, obviously we need to set the lora-related configurations in the `TransformerConfig` when initalizing the model about its mlp sub-layers. With that help, plus if the `BaseTrainer` above is designed well to take over many functionalities by default, then the subclass `LoraTrainer` might only have to overwrite a little bit of codes to just toggle only the lora weights to be trainable in initialization, and save only the lora weights into checkpoint directory when the configuration `save_only_lora` is set to `True`. If you are interested, you can check how `HF PEFT LoraModel` support lora fine-tuning with a complicated adapter design in [References](#references).
* Here's a full table that informs you the details about each configuration in the `BaseTrainConfig` dataclass as follows (*Note: "Required" means the argument must be provided with non-`None` values during initializationm and "Fixed" means the argument cannot be set during initialization and retain their default values.*):

    | **Config Name**                 | **Type**                      | **Default**                    | **Required** | **Fixed** | **Description**                                                                                                                                      |
    |----------------------------------|-------------------------------|--------------------------------|--------------|-----------|------------------------------------------------------------------------------------------------------------------------------------------------------|
    | `train_steps`                    | `int`                         | `None`                         | `True`       | `False`   | The number of training steps to run.                                                                                      |
    | `eval_interval`                  | `Optional[int]`               | `None`                         | `False`      | `False`   | The interval steps at which the model will be evaluated during training. If `None`, no periodic evaluation occurs.                                          |
    | `eval_steps`                     | `int`                         | `0`                            | `False`      | `False`   | The number of steps to perform evaluations, i.e. the number of batches the model needs to consume on the evaluation dataset in each evaluation.                                                       |
    | `shuffle`                        | `bool`                        | `False`                        | `False`      | `False`   | Whether to shuffle the training and evaluation dataset in initialization.                                                                                          |
    | `shuffle_seed`                   | `Optional[int]`               | `None`                         | `False`      | `False`   | The seed used for shuffling the dataset. If `None`, no specific seed is used.                                                                       |
    | `optimizer_type`                 | `OptimizerType`               | `OptimizerType.ADAMW`          | `False`      | `False`   | The type of optimization algorithm to use for training. Defaults to `OptimizerType.ADAMW` (*See the Pytorch optimizer modules in [References](#references) for more details*).                                                                  |
    | `learning_rate`                  | `float`                       | `None`                         | `True`       | `False`   | The static learning rate for the optimizer. This is a required field, often set in the range of `[1e-5, 1e-3]`.                                                                                     |
    | `momentum`                       | `float`                       | `0.0`                          | `False`      | `False`   | The momentum factor for the optimizer (*only used for SGD*).                                                                                        |
    | `betas`                          | `Tuple[float, float]`         | `(0.9, 0.999)`                 | `False`      | `False`   | The beta values for the Adam optimizer (*only used for ADAM & ADAMW*).                                                                               |
    | `weight_decay`                   | `float`                       | `0.0`                          | `False`      | `False`   | The weight decay factor for the optimizer (*only used for ADAMW*).                                                                                  |
    | `load_ckpt_dirs`                 | `Optional[Union[str, List[str]]]`| `None`                        | `False`      | `False`   | Directories to load pretrained checkpoints from. If `None`, no checkpoint loading occurs.                                                                      |
    | `load_ckpt_step`                 | `bool`                        | `True`                         | `False`      | `False`   | Whether to load the step index as well when loading the checkpoint. If does, the training step will begin with that step instead of `0` to resume training (*NOTE: the `train_steps` above indicates the number of training steps to run, instead of the maximum step index, thus whatever step the training starts from, it will last for `train_steps` steps unless other stopping criteria are met*).                                                                                         |
    | `save_interval`                  | `Optional[int]`               | `None`                         | `False`      | `False`   | The interval steps at which the model parameters will be saved as a checkpoint. If `None`, no periodic checkpoint saving occurs.                                                       |
    | `save_last_step`                 | `bool`                        | `True`                         | `False`      | `False`   | Whether to save the checkpoint for the last training step, regardless of the `save_interval`.                                                                                                 |
    | `save_ckpt_dir`                  | `str`                         | `"."`                          | `False`      | `False`   | The directory to save checkpoints. The directory will be created if it does not exist.                                                              |
    | `max_shard_size`                 | `int`                         | `1024`                         | `False`      | `False`   | The maximum shard size (in MB) for one single shard of checkpoint in `.safetensors` format.                                                                                              |
    | `step_idx_width`                 | `int`                         | `5`                            | `False`      | `False`   | The width of the step index in checkpoint directory name, e.g. if this checkpoint is for the step with the index `3` and the width is `5`, the checkpoint directory will be probably named as `step-00003`.                                                                                              |
    | `ckpt_step_prefix`               | `str`                         | `"step-"`                       | `False`      | `True`    | The prefix used in checkpoint directory name. Fixed value for consistency.                                                                               |
    | `ckpt_file_ext`                  | `str`                         | `"safetensors"`                | `False`      | `True`    | The file extension for checkpoint files, fixed to `.safetensors`.                                                                                                           |
    | `log_interval`                   | `Optional[int]`               | `None`                         | `False`      | `False`   | The interval at which the model's training state (*e.g., training loss, evaluation loss, learning rate if using dynamic learning rate scheduler*) will be logged to each logger (*e.g. terminal, wandb, tensorboard*). If `None`, no periodic logging occurs.                                      |
    | `log_last_step`                  | `bool`                        | `True`                         | `False`      | `False`   | Whether to log model's state for the last step, regardless of the `log_interval`.                                                                                                      |
    | `log_types`                      | `Tuple[TrainLogType]`          | `(TrainLogType.TERMINAL,)`      | `False`      | `False`   | The type(s) of logger(s) to log model's state to, e.g. if `log_types=(TrainLogType.TERMINAL, TrainLogType.WANDB, TrainLogType.TENSORBOARD)`, then the model's state will be logged to terminal, wandb, and tensorboard respectively.                                                                               |
    | `log_kwargs`                     | `dict`                        | `dict()`                       | `False`      | `False`   | Additional keyword arguments to pass to the specific loggers, e.g., `log_kwargs={"wandb_project": "lora_train", "tensorboard_log_dir": "./tensorboard/"}`.                                                                                      |
    | `device`                         | `str`                         | `"cpu"`                        | `False`      | `False`   | The device (`"cpu"` or `"cuda"`) where the training will be performed. |
* And as for the `LoRATrainConfig` as a sub-dataclass of `BaseTrainConfig`, we might only need to additionally provide specific (fixed) fields such as the patterns of lora weight names in the model's `state_dict`, and the boolean flag `save_only_lora` we've mentioned above.


#### Summary

In summary, you should implement several pytorch modules to support lora fine-tuning:
* `QADataset` is to prepare (training) dataset for Question-Answering task
* `ChatDataset` is to prepare (training) dataset for Chatbot task
* `BaseTrainer` is the basic trainer to support general training processes and functionalies
* `LoRATrainer` is the trainer inherited from `BaseTrainer` to adjust a little to simply support lora fine-tuning


#### Notice

* If you're not familar with the training configurations, please refer to the `HF Transformers Trainer` and `HF Transformers Training Arguments` in [References](#references) for more conceptual explanations and examples.
* We provide some helpful functions in `./src/utils.py`, including the load/save functions for `json`, `jsonl` and `safetensors` files. You can feel free to use them in your implementation.


#### References

*Hints: Here're some references which may be helpful to your task, or just deepen / broaden your knowledge to llm training:*

**!! Remember: it is a fundemental and essential capability to search, read, think and learn from the paper, source code, and official documentation for your answer, try NOT to rely too much on some biased and superficial blogs, e.g. CSDN !!**


* [Pytorch Training Loop](https://pytorch.org/tutorials/beginner/introyt/trainingyt.html#the-training-loop)
* [HF Transformers Trainer](https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/trainer#transformers.Trainer)
* [HF Transformers Training Arguments](https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/trainer#transformers.TrainingArguments)
* [HF Transformers Data Collator](https://huggingface.co/docs/transformers/v4.46.3/en/main_classes/data_collator)
* [HF Transformers PretrainedModel Save Pretrained Method](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/modeling_utils.py#L2677)
* [HF Safetensors Save File Method](https://github.com/huggingface/safetensors/blob/v0.4.5/bindings/python/py_src/safetensors/torch.py#L255)
* [HF Safetensors Save Model Method](https://github.com/huggingface/safetensors/blob/v0.4.5/bindings/python/py_src/safetensors/torch.py#L130)
* [HF PEFT LoraModel](https://github.com/huggingface/peft/blob/v0.13.2/src/peft/tuners/lora/model.py#L65)
* [Pytorch SGD Optimizer](https://pytorch.org/docs/stable/generated/torch.optim.SGD.html#sgd)
* [Pytorch Adam Optimizer](https://pytorch.org/docs/stable/generated/torch.optim.Adam.html#adam)
* [Pytorch AdamW Optimizer](https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html#adamw)
* [Python Rich Package Tutorial](https://realpython.com/python-rich-package/)
* [Pytorch Tensorboard SummaryWriter](https://pytorch.org/docs/stable/tensorboard.html#torch.utils.tensorboard.writer.SummaryWriter)
* [WanDB Tutorials](https://docs.wandb.ai/tutorials)


#### Bonus 🔥

* Besides of [LoRA](https://arxiv.org/pdf/2106.09685), there're a lot of other popular parameter-efficient fine-tuning strategies (*PEFT*), named a few of them: [Prompt Tuning](https://arxiv.org/pdf/2104.08691), [Prefix Tuning](Prefix-Tuning), [BitFit](https://arxiv.org/pdf/2106.10199), [KronA](https://arxiv.org/pdf/2212.10650), [RoSA](https://arxiv.org/pdf/2401.04679). For more comprehensive overview about PEFT, you can refer to some relative survey papers, such as [Delta Tuning](https://arxiv.org/pdf/2203.06904), [A Guide to PEFT](https://arxiv.org/pdf/2303.15647) and [LLM-Adapters](https://arxiv.org/pdf/2304.01933).
* Therefore, **as for the bonus for task3**, you can try to implement other trainer beyond lora, referring to the papers above as well as off-the-shelf implmentations in some awesome libraries, such as [HF PEFT](https://github.com/huggingface/peft/), [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory), etc. Or, you can choose some other training datasets and use whatever fine-tuning strategy to enhance the capability of the pretrained model on some domain-specific NLP tasks, such as code generation for some niche programming languages.
* **NOTE**: you might get into a lot of trouble like overfitting or the out-of-memory error, and you can feel free to pull out all the stops to tackle these problems, including but not limited to modifying the inner modules, switching to better base model, adjusting the training hyperparameters and using some external optimization tools, **as long as you still build this application upon the same code base in `src/`**.
