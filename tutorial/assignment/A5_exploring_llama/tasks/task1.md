### Task 1: Llama Model (plus bonus🔥)


#### TODO

You are required to implement a pytorch module named `LlamaModel` in `src/modeling/models/llama.py`.


#### Explanation

* Based on the `TransformerDecoderBlock` we've built in assignment4, we are going to build the `LlamaModel` module in this task, which mainly holds an instance of `TransformerDecoderBlock` as its backbone, and further provide some other functionality and APIs.
* As for its `forward` method, it takes `input_ids` and optional `cu_seqlens` as the first two arguments passing into the `TransformerDecoderBlock`, in order to apply the forward pass to get the output logits. Then, it applies a post-processing step towards this logits depending on the mode:
    * 1. `training mode`: it applies `cross_entropy` loss function to the logits and the given `labels` as the third optional argument, to get and return the scalar loss tensor. (*See the `Pytorch Cross Entropy Loss Functional` and `HF ForCausalLM Loss Function` in [References](#references) for more details about the loss function.*)
    * 2. `evaluation mode`: it applies row-wise `softmax` scaled with the fourth argument `temperature` to the logits, to get the next-token vocabulary probability distribution for each token, and **ONLY** returns the one of the last token for each sequence as the output vocabulary probability tensor with the shape of `[inferred_batch_size, vocab_size]`.
* Then, since all the modules we've built starting from assignment1 are filled with the initialized parameters and instantiated with the arbitrary configurations, this `LlamaModel` should provide the ability to load the actual configurations and pretrained parameters from the existing `Llama 3.2` model family (*See the model cards for the `Llama 3.2` light-weighted model family in [References](#references) for more details.*). Therefore, there're two extra APIs you should implement:
    * 1. `load_config`: this is a static method which takes a `config_file` argument as the path towards the `config.json` file inside any model directory of the `Llama 3.2` family, then parses the necessary configurations from this file to instantiate a `LlamaConfig` object. This `LlamaConfig` is a dataclass inherited from the `TransformerConfig` class, with some features disabled as `fixed field`, such as `MoE`-related configurations since the `Llama 3.2` family are all dense models. Furthermore, to provide some flexibility, this method also supports to pass in some extra configurations such as `device`, `seed`, etc through the optional `**extra_configs` keyword arguments.
    * 2. `load_parameters`: this is a member method which takes a `param_files` arguments as the single or multiple paths towards the `*.safetensors` files, containing the pretrained parameters inside any model directory of the `Llama 3.2` family, then parses the state dict from these files and loads the corresponding parameters into this `LlamaModel` module.
* Additionally, here are some trivial APIs inherited from the `TransformerDecoderBlock` module you should implement: `reset_parameters()`, `get_kv_cache()`, `set_kv_cache(kv_cache: TransformerDecoderKVCache)`, `reset_kv_cache()`, `num_parameters(learnable_only: bool = False, unit: str = "1")`, `num_memory_footprint(unit: str = "B")`. And you may just call the corresponding APIs from the `TransformerDecoderBlock` module instance you hold.


#### Summary

In summary, you should implement this `LlamaModel` module, which first of all maps and loads the configurations and the pretrained parameters from the existing `Llama 3.2` model family to instantiate itself. And then given `input_ids`, `cu_seqlens` (*optional*), `labels` (*optional*), `temperature` as inputs, it applies the forward pass to get the output logits from the inner `TransformerDecoderBlock` module, and finally applies a post-processing step depending on the mode, to return either the scalar loss tensor or the output vocabulary probability tensor for the last token for each sequence.


#### Notice

* When applying the forward pass in the `evaluation mode`, of-course we don't need to compute gradients, thus you can use `torch.no_grad()` context manager to disable the gradient computation. On the contrary, when applying the forward pass in the `training mode`, since the returned loss will be used to back-propagate the gradients, you should enable the gradient computation by either just removing `torch.no_grad()` or explicitly using `torch.enable_grad()` instead.
* The `inferred_batch_size` of the output vocabulary probability distribution is inferred from the `cu_seqlens` if provided (i.e. `inner_batch_size`), otherwise from `input_ids` (i.e. `batch_size`).
* The mapping from the existing `Llama 3.2` model family to the `LlamaModel` module, including the configurations and pretrained parameters, requires you some investigation into the original modeling codes and model repositories of the `Llama 3.2` family (*See the `Llama Config Dataclass`, `Llama for CausalLM Module`, or any model card of the `Llama 3.2` light-weighted model family in [References](#references) for more details.*).
* The basic introduction and usage about the `safetensors` file, as the popular persistence format for llm parameters, can be found in the `HF Safetensors Usage` in [References](#references).
* We've already collected all the light-weighted models from the `Llama 3.2` family and initialized their model repos with the official README files in the `../model/` directory, including `Llama-3.2-1B`, `Llama-3.2-1B-Instruct`, `Llama-3.2-3B` and `Llama-3.2-3B-Instruct`. As for the other necessary files such as the `config.json` and `*.safetensors`, you can find and download from either the official huggingface repo or the nju-box.
* We provide some helpful functions in `./src/utils.py`, including the load/save functions for `json`, `jsonl` and `safetensors` files. You can feel free to use them in your implementation.


#### References

*Hints: Here're some references which may be helpful to your task, or just deepen / broaden your knowledge to llm modeling:*

**!! Remember: it is a fundemental and essential capability to search, read, think and learn from the paper, source code, and official documentation for your answer, try NOT to rely too much on some biased and superficial blogs, e.g. CSDN !!**


* [Llama Config Dataclass](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/configuration_llama.py#L26)
* [Llama For CausalLM Module](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L1105)
* [HF Safetensors Usage](https://huggingface.co/docs/safetensors/index#usage)
* [Pytorch Cross Entropy Loss Functional](https://pytorch.org/docs/stable/generated/torch.nn.functional.cross_entropy.html#torch.nn.functional.cross_entropy)
* [HF ForCausalLM Loss Function](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/loss/loss_utils.py#L32)
* [Llama3 Paper](https://arxiv.org/pdf/2407.21783)
* [Llama 3.2 Lightweight Models Card](https://www.llama.com/docs/model-cards-and-prompt-formats/llama3_2/#-llama-3.2-lightweight-models-(1b/3b)-)
* [HF Llama3.2-1B Model Card](https://huggingface.co/meta-llama/Llama-3.2-1B)
* [HF Llama3.2-1B Instruct Model Card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct)
* [HF Llama3.2-3B Model Card](https://huggingface.co/meta-llama/Llama-3.2-3B)
* [HF Llama3.2-3B Instruct Model Card](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)


#### Bonus 🔥

* Considering the resources limitation of most of the students, we only take the light-weighted models of the `Llama 3.2` family as an example, to guide you to learn how to load the pretrained materials including configurations and parameters into your own model.
* In fact, besides of loading from larger models of the [Llama3.2 family](https://huggingface.co/collections/meta-llama/llama-32-66f448ffc8c32f949b04c8cf), there're a lot of other SOTA dense models whose architectures vary a little from the one of `Llama`, such as [GLM4 family](https://huggingface.co/collections/THUDM/glm-4-665fcf188c414b03c2f7e3b7), [Qwen2.5 family](https://huggingface.co/collections/Qwen/qwen25-66e81a666513e518adb90d9e), [Mistral family](https://huggingface.co/mistralai?search_models=mistral), etc, and thus their pretrained materials can also be loaded, just to replace the mapping logits in the loading methods. Furthermore, since we've extended to `MoE`-style of sparse MLP structure in assignment2, it is likely to load the pretrained materials from the SOTA sparse models such as [Mixtral family](https://huggingface.co/mistralai?search_models=mixtral) and even [DeepSeek-V2 family](https://huggingface.co/collections/deepseek-ai/deepseek-v2-669a1c8b8f2dbc203fbd7746).
* Therefore, **as for the bonus for task1**, you can give a try to implement some `XXXModel` module and load the corresponding pretrained materials from the existing `XXX` model family, just like what we've done to implement `LlamaModel`. And of-course, if you choose some "bad" model family with its specific features we haven't supported in the previous assignments, you may have to modify some inner modules to make sure it runs as expected.