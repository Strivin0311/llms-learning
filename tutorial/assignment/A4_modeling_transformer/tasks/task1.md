### Task 1: Transformer Decoder KVCache (20 points)

#### TODO

You are required to implement a simple auxiliary pytorch module named `TransformerDecoderKVCache` in `src/modeling/transformer.py`.


#### Explanation

* It is well acknowledged that most of the contemporary Transformer-based LLMs, such as `Llama`, `ChatGLM`, utilize decoder-only structure, pretrained with the causal language modeling (CLM) target, indicating that they have to follow the auto-regressive generation paradigm during inference token-by-token, which can be naturally divided into two stages:
    * 1. **Prefilling stage**: the LLM is fed into a whole unseen query sequence without attending to one another before, then it applies forward pass and returns the predicted next-token probability distribution, from which we can sample to generate the first token.
    * 2. **Decoding stage**: then to generate the subsequent tokens, every time the newly generated token is fed back into the LLM as a single query, which needs to be attended to the keys and applied to the values of itself and the previous tokens (including the ones of the original query and the prior generated tokens).
* So to avoid the re-computation of the keys and values of the previous tokens for each transformer decoder layer (*See [Task2](./task2.md)*) during the decoding stage, we can cache them starting from the prefilling stage, and retrieve and reuse the key-value tensors from the cache whenever a new generated token is applied, with updating them along the sequence dimension to prepare for the next one.
* To better manage this cache to store, retrieve and update past key-value tensors, here we are going to design a simple module as a data structure called `TransformerDecoderKVCache`, with the API reference for you to implement as follows:
    * `__init__(qkv_layout, num_layers=1)`: initialize the cache given the `qkv_layout` (to deduce the kv shape), and the optional `num_layers` (to be pre-aware of the number of layers for some convenience, e.g. you can pre-allocate some inner data as your design, though it might be unnecessary).
    * `has(layer_idx)`: check if the cache has the key-value tensors for a specific layer.
    * `get(layer_idx)`: get the key-value tensors for a specific layer, with the cumulative sequence lengths if using varlen attention with `qkv_layout=AttnQKVLayout.THD`, otherwise return `None` as a placeholder.
    * `set(layer_idx, k, v, cu_seqlens=None)`: set the key-value tensors for a specific layer (*if already exists, overwrite them*), with the optional cumulative sequence lengths if using varlen attention with `qkv_layout=AttnQKVLayout.THD`.
    * `append(layer_idx, k, v, cu_seqlens=None)`: update the existing cache with the given key-value tensors along the sequence dimension for a specific layer (*if not exists, this api should work like `set` instead*), with the optional given cumulative sequence lengths if using varlen attention with `qkv_layout=AttnQKVLayout.THD`.
    * `reset()`: clear the cache memory and reset to the initial state.
* Of-course, the data stucture above is a simple and naive implementation (*See the HF transformers implementation in the [Reference](#references)*), and there are a lot of more elaborate designs to reduce the kv cache memory footprint and improve the decoding efficiency for real-world practice (*See the relative vLLM documentations in the [Reference](#references)*).

#### Summary

In summary, you should implement this `TransformerDecoderKVCache` module following the API reference above, which will be used in subsequent tasks as an auxiliary module to manage the kv cache in a simple way for each transformer decoder layer during inference.


#### Notice

* All of the arguments passed to `set` and `append` are ensured to be consistent with the existing ones, for example, the `cu_seqlens` will be provided if the `qkv_layout` is `AttnQKVLayout.THD`, and the meta info of the pass-in tensors like dtype, device, and the inner batch size inferred from the `cu_seqlens`, will also be the same as the ones in the cache respectively. But it's still a good habit to check the consistency of the arguments for the sake of error handling and ensuring the correctness of the cache.
* Neither time nor space complexity is required in this task, thus you are free to design the data structure in any way you like, as long as it is correct and affordable.


#### References

*Hints: Here're some references which may be helpful to your task, or just deepen / broaden your knowledge to transformer decoder kv cache:*

**!! Remember: it is a fundemental and essential capability to search, read, think and learn from the paper, source code, and official documentation for your answer, try NOT to rely too much on some biased and superficial blogs, e.g. CSDN !!**


* [HF Transformers Dynamic Cache Module](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/cache_utils.py#L351)
* [vLLM Paged Attention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html)
* [vLLM Chunked Prefill](https://docs.vllm.ai/en/latest/models/performance.html)
* [vLLM Automatic Prefix Caching](https://docs.vllm.ai/en/latest/automatic_prefix_caching/apc.html)
* [vLLM Fp8 E4M3 KV Cache](https://docs.vllm.ai/en/latest/quantization/fp8_e4m3_kvcache.html)