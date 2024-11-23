

### Task 2: Parallel Vocab Embedding (30 points)

#### TODO

You are required to implement a pytorch module called `ParallelVocabEmbedding` in `src/modeling/vocab_emb.py`.

#### Explanation

* Given a vocabulary table with the size `vocab_size` denoted as `v`, the vocabulary embedding layer in the transformer displays as a standard pytorch embedding module, which takes the input ids with the shape `[batch_size, seq_len]` as input (*denoted as I, with the shape `[b, s]`*), dispatchs the corresponding embedding vector for each id within the range `[0, ..., v-1]`, and returns the embedding tensor with the shape `[batch_size, seq_len, emb_size]` (*denoted as E, with the shape `[b, s, e]`*), by looking up a learnable embedding table (*denoted as T, with the shape `[v, e]`*).

* However, with the development of LLMs, the vocabulary table size scales rapidly up to `128K+`, which makes the embedding table too large to fit in one single GPU.

* Therefore, we have to implement a "parallel vocab embedding module" to solve this problem, by sharding the embedding table into equal partitions in a process group with the size `world_size` denoted as `w`, and each rank with the rank idx `r` in the range of `[0, w - 1]` in that process group only handles one partition of it with the size `n = v // w` (*after which the full vocabulary embeddings can be reduced together by communication, but this is out of the scope of this assignment thus we skip this process*). 

* In this way, not only can we reduce the size of the embedding table in each single GPU, but also we are able to apply the large vocabulary embedding operation in parallel to speed up.

* Specifically, given a full vocabulary size `v`, embedding size `e`, current rank index `r`, and world size `w`:

    * 1. The `ParallelVocabEmbedding` first initializes a partial embedding table `Tr` from a **normal distribution**, focusing **only** on the ids within the range `[r * n, (r + 1) * n - 1]` (denoted as `R`).
    * 2. Next, it takes input `I`, retrieves the embedding vectors from `Tr` for ids within `R`, and replaces any out-of-range ids with all-zero vectors.
    * 3. Finally, it returns an incomplete embedding tensor `Er`, shaped as `E` but **ONLY** contains embedding vectors for the ids in `R`, leaving the rest as "holes." (*This allows the full vocabulary embeddings to be reconstructed by summing the incomplete embeddings from all ranks, due to their "orthogonality."*)

* By the way, you should implement the `reset_parameters` member method for this `ParallelVocabEmbedding` module class as well, to initialize the `Tr` from a **normal distribution** given the mean (*denoted as `init_mean`*) like `0.`, standard deviation (*denoted as `init_std`*) like `1.`, and base seed (*denoted as `init_base_seed`*) like `42` (**NOTE: the real random seed will be `init_base_seed + r`, to avoid the same initialization for all ranks, which is also a standard approach in distributed DL settings**).


#### Summary

In summary, you are supposed to implement a pytorch module `ParallelVocabEmbedding`, which firstly initializes the partial embedding table `Tr` from a normal distribution controlled by `init_mean`, `init_std` and `init_base_seed`, then takes the input `I` and returns an incomplete embedding tensor `Er` (*not `E`*).


#### Notice

* The `dtype` of input ids `I` is ensured to be `torch.long`.
* Your implementation should **NOT** change `I` in-place, since it may have other usages in other places.
* The `dtype` and `device` in arguments are for the learnable embedding table `Tr`, which may be different from the ones of `I`.
* The returned `Er` should shares the same device with `I`, and the same dtype as `Tr`.
* The `v` is ensured to be divisible by `w` in all test cases, but it's still a good habit to check the divisibility in the `__init__` method.
* The `reset_parameters` method should be automatically called once in the `__init__` method when you initialize the parameters.
* You can of course refer to how `ChatGLM` and `Megatron` build their (parallel) vocab embedding modules in [References](#references), but be careful about the specific requirments above, which differs a little from the ones of neither `ChatGLM` nor `Megatron`.


#### References

*Hints: Here're some references which may be helpful to your task, or just deepen / broaden your knowledge to vocabulary embedding layers in deep learning:*

**!! Remember: it is a fundemental and essential capability to search, read, think and learn from the paper, source code, and official documentation for your answer, try NOT to rely too much on some biased and superficial blogs, e.g. CSDN !!**

* [Pytorch Embedding Module](https://pytorch.org/docs/stable/generated/torch.nn.Embedding.html)
* [Pytorch Embedding Functional](https://pytorch.org/docs/stable/generated/torch.nn.functional.embedding.html)
* [ChatGLM Vocab Embedding Module](https://huggingface.co/THUDM/chatglm3-6b/blob/main/modeling_chatglm.py#L706)
* [Megatron Vovab Parallel Embedding Module](https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/tensor_parallel/layers.py#L156)
* [Pytorch Normal Initialization](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.normal_)
