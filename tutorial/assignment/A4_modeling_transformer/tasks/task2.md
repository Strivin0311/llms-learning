### Task 2: Transformer Decoder Layer

#### TODO

You are required to implement a pytorch module named `TransformerDecoderLayer` in `src/modeling/transformer.py`.


#### Explanation

* The architecture of a decoder-only transformer-based LLM can be metaphorically referred to as a "giant stacked hamburger":
    * 1. The top / bottom slice of the bread represents the input / output embedding layer in charge of the transformation between the input / output token space and the latent hidden space respectively
    * 2. The central stacked beef patty consists of a sequence of decoder layers that process the hidden states in the latent hidden space with the self-attention mechanism (*for inter-token interaction*) and the mlp mechanism (*for intra-token linear plus non-linear transformation*)
* Therefore, the target of this task is to make one piece of the stacked beef patty elaborately like a chief, the `TransformerDecoderLayer` module, and in the later task, we will stack multiple of them together covered with the "bread" to form this "monster hamburger" (*See [Task3](./task3.md)*).
* One transformer decoder layer consists of two main sub-layers:
1. self-attention layer: given the input hidden states $X$ with the shape: `[batch_size, seq_len, hidden_size]` (*if `cu_seqlens` is also provided, then the `batch_size` here is ensured to be `1`, while the real batch size is inferred from `cu_seqlens`, i.e. `inner_batch_size`, since the inner sequences are concatenated along the `seqlen` dim.*), as shown in the equation below, this layer mainly projects the normalized input hidden states $\tilde{X}$ to $Q, K, V$, assigns $Q,K$ the positional embeddings to get embedded $\tilde{Q}, \tilde{K}$, and applies the self-attention operation given $\tilde{Q}, \tilde{K}, V$ to get the attention output $\tilde{O}$, which is then projected back to hidden space and added with the residual $R$ to get the final output hidden states $O$.
         
$$
\begin{aligned}
& R = X \\
& \tilde{X} = \text{Norm}(X) \\
& Q, K, V = \text{split}(\tilde{X}\times W_{QKV}) \\
& \tilde{Q}, \tilde{K} = \text{RoPE}(Q), \text{RoPE}(K) \\
& \tilde{O} = \text{SelfAttn}(\tilde{Q}, \tilde{K}, V) \\
& O = \tilde{O}\times W_{O} + R \\
\end{aligned}
$$


2. mlp layer: given the input hidden states $X$ (equal to the output hidden states $O$ from the above self-attention layer), this layer mainly applies the mlp transformation to the normalized input hidden states $\tilde{X}$ with residual connection to get the final output hidden states $O$.
    
$$
\begin{aligned}
& R = X \\
& \tilde{X} = \text{Norm}(X) \\
& \tilde{O} = \text{MLP}(\tilde{X}) \\
& O = \tilde{O} + R \\
\end{aligned}
$$

* To fully utilize the modules we've built in the previous assignments, here we implement the $\text{Norm}$ as `GroupRMSNorm`, $\text{RoPE}$ as `NTKAwareRoPE`, $\text{SelfAttn}$ as `OfflineSlidingWindowAttn` or `OnlineSlidingWindowAttn`, $\text{MLP}$ as `DenseMLPWithLoRA` or `SparseMLPWithLoRA`.
* To support forward pass during inference especially in the decoding phase, the `forward` method of the `TransformerDecoderLayer` module also supports the optional `kv_cache` argument instantiated by `TransformerDecoderKVCache`(*See [Task1](./task1.md)*), which manages the kv cache of all the decoder layers, and you should retrieve the ones of this particular decoder layer, attend the current query with not only the current key-value but also the past key-value in the cache, and update the cache in-place by calling the appropriate APIs we've implemented in [Task1](./task1.md).
* In the meanwhile, the attention mask should be aligned accurately when attending the current query with the past key-value. Recall that for the task2 of assignment3, we've already **aligned the mask to the bottom-right part** when the "sequence"-dim are different between the query and the key-value following the flash-attention's settings (*See the Flash Attention Interface in [References](#references) for more examples*). Thus this issue is natually handled by the attention sub-module in the decoding phase during inference since the current query always has the largest position index than the ones of the past key-value, i.e. it corresponds the last row of the latent square attention mask matrix.
* Another relevant issue is to assign the correct positional embeddings for the current query since it may be a single token with the larget position index instead of `0`. **Therefore, we introduce a new optional argument `offset: int=0` for the `forward` method of the `NTKAwareRoPE` module for you to modify a little of your own code to additionally support shifting all the positional indexs by a fixed offset for the given input tensor, i.e. the orignal index range [0, seq_len-1] becomes [offset, offset+seq_len-1].** Of-course, we do not validate this new feature in this task, so you can always ignore it with your old implementation of the `NTKAwareRoPE` module, and come up with another solution to deal with the position index issue accurately.
* To manage the initialization of the `TransformerDecoderLayer` module and its sub-modules, we provide a general config dataclass `TransformerConfig` in `src/modeling/transformer.py`, in which you can find all the necessary arguments to initialize the `TransformerDecoderLayer` module (*except for the `layer_idx` in the range of `[0, config.num_layers]`, which is an optional manually-set argument to specify the index of the current decoder layer*) and its sub-modules. 
* Here's a full table that informs you the details about each configuration in the `TransformerConfig` dataclass as follows (*Note: "Required" means the argument must be provided with non-`None` values during initializationm and "Fixed" means the argument cannot be set during initialization and retain their default values.*):

   | **Config Name**            | **Type**                     | **Default**                   | **Required** | **Fixed** | **Description**                                                                                                                                       |
   |----------------------------|------------------------------|-------------------------------|--------------|-----------|-------------------------------------------------------------------------------------------------------------------------------------------------------|
   | `num_layers`               | `int`                        | `None`                        | `True`       | `False`   | The number of transformer decoder layers used in `TransformerDecoderBlock` (*See [Task3](./task3.md)*).                                                                                                        |
   | `hidden_size`              | `int`                        | `None`                        | `True`       | `False`   | The dimension of the hidden states.                                                                                        |
   | `ffh_size`                 | `int`                        | `None`                        | `True`       | `False`   | The dimmension of the intermediate hidden states used in `DenseMLPWithLoRA` and `SparseMLPWithLoRA`.                                                                                               |
   | `max_seq_len`              | `int`                        | `None`                        | `True`       | `False`   | The maximum sequence length used in `NTKAwareRoPE` and `OnlineSlidingWindowAttn`.                                                                                                             |
   | `param_dtype`              | `torch.dtype`                | `torch.float32`               | `False`      | `False`   | The data type of **ALL** the parameters.                                                                                                                       |
   | `param_device`             | `str`                        | `"cpu"`                       | `False`      | `False`   | The device on which **ALL** of the parameters are located.                                                                                                       |
   | `init_base_seed`           | `int`                        | `42`                          | `False`      | `False`   | The basic random seed for parameter initialization.                                                                                                               |
   | `rank`                     | `int`                        | `0`                           | `False`      | `True`    | The rank of the process, fixed to `0`.                                                                               |
   | `world_size`               | `int`                        | `1`                           | `False`      | `True`    | The number of processes, fixed to `1`.                                                                         |
   | `process_group`            | `Optional[ProcessGroup]`     | `None`                        | `False`      | `True`    | The process group for distributed training, fixed to `None`.                                                            |
   | `vocab_size`               | `int`                        | `None`                        | `True`       | `False`   | The size of the vocabulary used in `ParallelVocabEmbedding` and `lm_head` layer (*See [Task3](./task3.md)*).                                                                                                             |
   | `vocab_init_mean`          | `float`                      | `0.0`                         | `False`      | `False`   | The mean value of the normal distribution to initialize the vocabulary embedding table in `ParallelVocabEmbedding`.                                                                                                 |
   | `vocab_init_std`           | `float`                      | `1.0`                         | `False`      | `False`   | The standard deviation of the normal distribution to initialize the vocabulary embedding table in `ParallelVocabEmbedding`.                                                                                   |
   | `rope_base`                | `int`                        | `10000`                       | `False`      | `False`   | The base value to control the frequences in `NTKAwareRoPE`.                                                                                                 |
   | `rope_ratio`               | `int`                        | `1`                           | `False`      | `False`   | The scaling ratio to extraplolate the frequencies used in `NTKAwareRoPE`.                                                                                                   |
   | `rope_dynamic`             | `bool`                       | `False`                       | `False`      | `False`   | Whether to dynamically update cached cos/sin embeddings in `NTKAwareRoPE`.                                                                                              |
   | `group_size`               | `Optional[int]`              | `None`                        | `False`      | `False`   | The group size to split the hidden size in `GroupRMSNorm`.                                                                                                                      |
   | `eps`                      | `float`                      | `1e-5`                        | `False`      | `False`   | The epsilon value to avoid numerical instability in `GroupRMSNorm`.                                                                                                                   |
   | `norm_init_range`          | `tuple`                      | `(-1.0, 1.0)`                 | `False`      | `False`   | The range of the uniform distribution to initialize the scaling parameters in `GroupRMSNorm`.                                                                                                      |
   | `proj_init_seed`           | `int`                        | `42`                          | `False`      | `False`   | The random seed to initialize projection matrices, including `qkv_proj`, `o_proj`, as well as the `lm_head` if `lm_head_tied=False`.                                                                                                   |
   | `proj_init_mean`           | `float`                      | `0.0`                         | `False`      | `False`   | The mean value of the normal distribution to initialize projection matrices.                                                                                                     |
   | `proj_init_std`            | `float`                      | `1.0`                         | `False`      | `False`   | The standard deviation of the normal distribution to initialize projection matrices.                                                                                       |
   | `lm_head_tied`             | `bool`                       | `False`                       | `False`      | `False`   | Whether to tie the weights of the `lm_head` layer to the one of the vocab embedding layer (*See [Task3](./task3.md)*).                                                                                                                |
   | `online_attn_block_size`   | `Optional[int]`              | `None`                        | `False`      | `False`   | The block size for `OnlineSlidingWindowAttn`. If `None`, use `OfflineSlidingWindowAttn` instead.                                              |
   | `head_dim`                 | `int`                        | `None`                        | `True`       | `False`   | The dimension of each attention head.                                                                                                    |
   | `num_q_head`               | `int`                        | `None`                        | `True`       | `False`   | The number of query heads.                                                                                                               |
   | `num_kv_head`              | `int`                        | `None`                        | `True`       | `False`   | The number of key/value heads.                                                                                                           |
   | `qkv_pack_format`          | `AttnQKVPackFormat`          | `AttnQKVPackFormat.Q_K_V`     | `False`      | `False`   | The packing format for QKV tensors.                                                                                                       |
   | `qkv_layout`               | `AttnQKVLayout`              | `AttnQKVLayout.BSHD`          | `False`      | `False`   | The shape layout for QKV tensors.                                                                                                                            |
   | `window_size`              | `Optional[int]`              | `None`                        | `False`      | `False`   | The window size for sliding window attention.                                                                                                                         |
   | `causal`                   | `bool`                       | `False`                       | `False`      | `False`   | Whether to apply causal mask to the attention.                                                                                                             |
   | `softmax_dropout_rate`     | `float`                      | `0.0`                         | `False`      | `False`   | The dropout rate applied after the softmax operation.                                                                                                 |
   | `softmax_dropout_seed`     | `int`                        | `42`                          | `False`      | `False`   | The random seed for softmax dropout.                                                                                                                   |
   | `softmax_scale`            | `Optional[float]`            | `None`                        | `False`      | `False`   | The scaling factor applied to the softmax logits.                                                                                                  |
   | `softmax_cap`              | `Optional[float]`            | `None`                        | `False`      | `False`   | The capping value to apply `softmax capping` to adaptively control the magnitude of the softmax logits, if `None`, use `softmax temperature` trick instead.                                                                                                              |
   | `softmax_temp`             | `float`                      | `1.0`                         | `False`      | `False`   | The temperature value to apply `softmax temperature` to control the sharpness of the softmax distribution when `softmax capping` is disabled.                                                                                                 |
   | `softmax_clip_range`       | `Tuple[float, float]`        | `(0.0, 1.0)`                  | `False`      | `False`   | The clipping range to apply `softmax clipping` to prevent the outliers in the softmax weights.                                                                                                            |
   | `apply_qk_norm`            | `bool`                       | `False`                       | `False`      | `False`   | Whether to apply `QK layer normalization` to the query and key tensors.                                                                                         |
   | `qk_norm_group_size`       | `Optional[int]`              | `None`                        | `False`      | `False`   | The specific group size for `QK layer normalization` if enabled. Other configurations for `QK layer normalization` share the same as above.                                                              |
   | `activation_type`          | `MLPActivationType`          | `MLPActivationType.SILU`      | `False`      | `False`   | The activation function type used in the mlp layer.                                                                                           |
   | `lora_rank`                | `int`                        | `0`                           | `False`      | `False`   | The rank for LoRA.                                                                                                              |
   | `lora_alpha`               | `Optional[float]`            | `None`                        | `False`      | `False`   | The alpha parameter for LoRA.                                                                                                                          |
   | `lora_dropout_rate`        | `float`                      | `0.0`                         | `False`      | `False`   | The dropout rate for LoRA layers.                                                                                                                      |
   | `lora_dropout_seed`        | `int`                        | `42`                          | `False`      | `False`   | The random seed for LoRA dropout.                                                                                                                      |
   | `lora_init_base_seed`      | `int`                        | `42`                          | `False`      | `False`   | The base random seed to initialize the parameters of LoRA.                                                                                                    |
   | `num_experts`              | `Optional[int]`              | `None`                        | `False`      | `False`   | The number of experts for `SparseMLPWithLoRA`. If `None`, then use `DenseMLPWithLoRA` instead.                                                              |
   | `moe_topk`                 | `int`                        | `1`                           | `False`      | `False`   | The top-k value for expert routing in `SparseMLPWithLoRA`.                                                                                                         |
   | `gate_init_mean`           | `float`                      | `0.0`                         | `False`      | `False`   | The mean value of the normal distribution to initialize the gating parameters.                                                                                             |
   | `gate_init_std`            | `float`                      | `1.0`                         | `False`      | `False`   | The standard deviation of the normal distribution to initialize the gating parameters.                                                                               |



#### Summary

In summary, you should implement this `TransformerDecoderLayer` module, which takes a hidden states tensor $X$ with cumulative sequence lengths $cu\_seqlens$ and a transformer kv cache $kv\_cache$ optionally as inputs, applies an offline / online self-attention sub-layer and a dense / sparse mlp sub-layer sequentially with group rms normalization, linear projection and residual connection, and finally returns the output hidden states tensor $O$ with the same shape as $X$, as the input for the next decoder layer.

#### Notice

* To assign each sub-module or sub-operation of each decoder layer an unique random seed, as usuall, we add some offsets to the basic ones from the `TransformerConfig` to set the actual random seed, as listed in the following table:

| Sub-module or Sub-Operation | Basic Random Seed | Offset |
| --------------------------- | ----------------- | ------ |
| `qkv_proj` in the `i`-th decoder layer | `config.proj_init_seed` | `i + 1` |
| `o_proj` in the `i`-th decoder layer | `config.proj_init_seed` | `i + 2` |
| `attn_norm` in the `i`-th decoder layer | `config.init_base_seed` | `i + 1` |
| `attn` in the `i`-th decoder layer | `config.init_base_seed` | `i + 2` |
| `mlp_norm` in the `i`-th decoder layer | `config.init_base_seed` | `i + 3` |
| `mlp` in the `i`-th decoder layer | `config.init_base_seed` | `i + 4` |
| `softmax_dropout` in the `i`-th decoder layer | `config.softmax_dropout_seed` | `i` |
| `lora` in the `i`-th decoder layer | `config.lora_init_base_seed` | `i` |
| `lora_dropout` in the `i`-th decoder layer | `config.lora_dropout_seed` | `i` |
| `vocab_embed` in the decoder block | `config.init_base_seed` | `0` |
| `lm_head` in the decoder block | `config.proj_init_seed` | `0` |
| `final_norm` in the decoder block | `config.init_base_seed` | `0` |

* We guarantee to feed the decoder layer with the inputs consistent with the `qkv_layout` and the optional `cu_seqlens` (*including those stored in the `kv_cache`*) as well. But it is still a good habit to check the consistency of the arguments for the sake of error handling and ensuring the correctness of the decoder layer.

* We guarantee to only use the `OnlineSlidingWindowAttn` when the conditions below are all met:
    * ① the sequence length equals to `max_seq_len`;
    * ② the `kv_cache` is `None`;
    * ③ the `qkv_layout` is `AttnQKVLayout.BSHD` and thus the `cu_seqlens` is `None`;
    * ④ the `qkv_pack_format` is `AttnQKVPackFormat.Q_K_V`.

* The meta attributes of the input hidden states like dtype and device might be different from the ones of the parameters, hence you should deal it carefully and make sure the meta attributes of the output hidden states are consistent with the ones of the input hidden states.

#### References

*Hints: Here're some references which may be helpful to your task, or just deepen / broaden your knowledge to transformer decoder layer:*

**!! Remember: it is a fundemental and essential capability to search, read, think and learn from the paper, source code, and official documentation for your answer, try NOT to rely too much on some biased and superficial blogs, e.g. CSDN !!**


* [Llama MLP Module](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L229)
* [Llama Attention Module](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L275)
* [Llama DecoderLayer Module](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L626)
* [Flash Attention Interface](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py)
