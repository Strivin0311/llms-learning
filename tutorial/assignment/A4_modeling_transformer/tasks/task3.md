### Task 3: Transformer Decoder Block (30 points)


#### TODO

You are required to implement a pytorch module named `TransformerDecoderBlock` in `src/modeling/transformer.py`.


#### Explanation

* Based upon the explanation in [Task2](./task2.md), we continue to implement the `TransformerDecoderBlock` module, stacking multiple of `TransformerDecoderLayer` modules covered with the input embedding layer instantiated by the `ParallelVocabEmbedding` module and the output embedding layer instantiated by the standard `nn.Linear` module.
* Following Llama's design (*See Llama Model Module in [References](#references) for more details*), as for the input embedding layer $\text{VocabEmb}$, it takes a sequence of token ids as the input tensor $I$ with the shape of `[batch_size, seq_len]`, and projects them from the vocabulary space to the hidden space as the initial hidden states $X_{ini}$ with the shape of `[batch_size, seq_len, hidden_size]`. Then, it is supposed to pass through the $L$ stacked decoder layers $\text{DecoderLayers}$ in a for-loop to get the final hidden states $X_{fin}$. As for the output embedding layer $\text{LMHead}$, it takes the final hidden states $\widetilde{X_{ini}}$, which is normalized by the $\text{FinalNorm}$, an instance of the `GroupRMSNorm` module, projects them from the hidden space back to the vocabulary space and outputs the vocaburary logits for each token as a tensor $Logits$ with the shape of `[batch_size, seq_len, vocab_size]`.
* The whole forward pass of the decoder block can be formalized as the following equation:

$$
\begin{aligned}
& X_{ini} = \text{VocabEmb}(I) \\
& X_{fin} = \text{DecoderLayers}\_{L}(X_{ini}) \\
& \widetilde{X_{ini}} = \text{FinalNorm}(X_{fin}) \\
& Logits = \text{LMHead}(\widetilde{X_{ini}})
\end{aligned}
$$

* Besides for the sub-layers, the decoder block is also supposed to initialize and hold an instance of the `TransformerDecoderKVCache` module, managing the kv cache of all the decoder layers and passed to each layer in each iteration of the for-loop **ONLY** during inference, i.e., when `self.training` is set to `False`.

* The decoder block module takes one instance of the `TransformerConfig` dataclass as the initialization argument and shares it to each decoder layer with its own layer index as the global constant configurations. And the detailed description of each configuration provided in the `TransformerConfig` dataclass can be found in [Task2 Explanation](./task2.md#explanation).

* Additionally, you should implement several simple but convenient APIs to access the kv cache and provide statistics on model parameters:
    * 1. `get_kv_cache()`: just return the kv cache object
    * 2. `set_kv_cache(kv_cache: TransformerDecoderKVCache)`: just set the kv cache object
    * 3. `reset_kv_cache()`: just call the `reset()` API of the kv cache object
    * 4. `num_parameters(learnable_only: bool = False, unit: str = "1")`: return the number of parameters in the given number-unit format (*choosing from "1", "K", "M", "B"*), and only count the learnable ones if `learnable_only` is set to `True`
    * 5. `num_memory_footprint(unit: str = "B")`: return the memory footprint of the parameters in the given byte-unit format (*choosing from "B", "KB", "MB", "GB"*)


#### Summary

In summary, you should implement this `TransformerDecoderBlock` module, which takes the token ids $I$ as inputs and returns the vocaburary logits tensor $Logits$ as outputs. It also takes charge of the kv cache of all the decoder layers and provides some APIs for the user to access the kv cache and provide statistics on model parameters.


#### Notice

* The random seed settings of the sub-layers of the decoder block can be found in [Task2 Notice](./task2.md#notice).

* The device of the input ids might be different from the one on which the parameters are located, hence you should deal it carefully and make sure the device of the output logits is consistent with the one of the input ids.

* There is a particular boolean configuration `lm_head_tied` in the `TransformerConfig` dataclass, indicating whether to tie the weights of the `lm_head` layer to the one of the vocab embedding layer, instead of initializing it separately (*See HF PretraiedModel Tie Weights in [References](#references) for more details*).


#### References

*Hints: Here're some references which may be helpful to your task, or just deepen / broaden your knowledge to transformer decoder block:*

**!! Remember: it is a fundemental and essential capability to search, read, think and learn from the paper, source code, and official documentation for your answer, try NOT to rely too much on some biased and superficial blogs, e.g. CSDN !!**


* [Llama Model Module](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L830)
* [Llama PretrainedModel Init Weights](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L739)
* [HF PretraiedModel Tie Weights](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/modeling_utils.py#L1915)