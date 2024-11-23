### Task 2: Online Sliding-Window Attention (40 points)

#### TODO

You are required to implement a pytorch module named `OnlineSlidingWindowAttn` in `src/modeling/attention.py`.


#### Explanation

* Building upon the `OfflineSlidingWindowAttn` module described in [task1](./task1.md), we continue to implement the `OnlineSlidingWindowAttn` module, which is the online version of the former one, only applying attention on a block of $Q_{bq_i},K_{bkv_j},V_{bkv_j}$ in `AttnQKVLayout.BSHD` layout and `AttnQKVPackFormat.Q_K_V` packing format, and aggregate the local output $O_{bq_i}^{(bkv_j)}$ of this block to the global output $O$, with the help of `log-sum-exp`-style softmax calibration coefficient $lse$.
* To be more specific, although both the computation cost and the memory footprint of the `attention` operation generally follow the quadratic complexity, we can reduce the memory complexity to almost linear by transforming the `offline softmax` to `online softmax` (*See the Online Softmax Paper in [References](#references)*). The basic idea is to split the `sq`-dim and `skv`-dim of $Q$ and $K,V$ equally to `bq`-dim and `bkv`-dim respectively as blocks, and each time only apply attention on a single block of $Q_{bq_i},K_{bkv_j},V_{bkv_j}$, where the indices $bq_i \in [0, \frac{sq}{bq}]$, $bkv_j \in [0, \frac{skv}{bkv}]$. 
* The local attention output of this block is denoted as $O_{bq_i}^{(bkv_j)}$, with the shape `[b, bq, hq, hd]`. Give the global output buffer $O$ with the shape `[b, sq, hq, hd]`, how can we aggregate $O_{bq_i}^{(bkv_j)}$ to $O$ accurately since the local/global softmax weights are not normalized from the same factors?
* As the stable softmax factorization equation shown below, if we split a row vector $X \in \mathbb{R}^{n}$ into two parts $X_1 \in \mathbb{R}^{n_1}$ and $X_2 \in \mathbb{R}^{n_2}$, where $n_1 + n_2 = n$, then the key to restore the softmax of the whole $X$ from the local softmax of $X_1$ and $X_2$ is to re-calculate the new normalization factor $l$ and new maximum value $m$.

$$
\begin{align}
&\text{softmax}(X) = \text{softmax}([X_1, X_2]) = \cfrac{\exp(X - m)}{l} = \left[ c_1 \cdot \text{softmax}(X_1), c_2 \cdot \text{softmax}(X_2)\right] = \left[ c_1 \cdot \cfrac{\exp(X_1 - m_1)}{l_1}, c_2 \cdot \cfrac{\exp(X_2 - m_2)}{l_2}\right], \\
&\text{where} \space c_i = \cfrac{l_i\cdot \exp(m_i-m)}{l}, \space m := \max{(X)} = \max{(m_1, m_2)}, \space l := \sum\exp(X - m) = \sum\exp(m_i - m) \cdot l_i, \space i\in \{1,2\}
\end{align}
$$

* To simplify the above calibration of softmax, we can also utilize the `log-sum-exp` operator $\text{lse}$ (*See the Pytorch LSE Functional in [References](#references)*) following the flash-attention's strategy (*See the Flash Attention 2 Paper in [References](#references) for more details*) to rewrite the stable softmax operation as:

$$
\begin{align}
&\text{softmax}(X) = \cfrac{\exp(X - m)}{\text{sum}(\exp(X - m))} = \cfrac{\exp(X - m)}{\exp(\log(\text{sum}(\exp(X - m))))} \\
&= \cfrac{\exp(X - m)}{\exp(\text{lse}(X - m))} = \exp(X - m - \text{lse}(X - m)) \\
&= \exp(X - (m + \text{lse}(X - m))) = \exp(X - \text{lse}(X))
\end{align}
$$

* where the last step uses a property of $\text{lse}$: $\text{lse}(X) = \max{(X)} + \text{lse}(X - \max{(X)})$ (*See the LSE Wiki in [References](#references)*). So the stable softmax factorization can be also re-formulated with the $\text{lse}$ operation as:

$$
\begin{align}
&\text{softmax}(X) = \text{softmax}([X_1, X_2]) = \exp(X - lse) = \left[ c_1 \cdot \text{softmax}(X_1), c_2 \cdot \text{softmax}(X_2)\right] \\
&= \left[ c_1 \cdot \exp(X_1 - lse_1), c_2 \cdot \exp(X_2 - lse_2)\right], \quad \text{where} \space c_i = \exp(lse_i - lse), \space i\in \{1,2\}, \space\text{and} \\
&lse := \text{lse}(X) = \log(\exp(lse_1) + \exp(lse_2)) = lse_{1} + \log(1 + \exp(lse_{2} - lse_{1})) \\
&\quad\space= lse_{max} + \log(1 + \exp(lse_{min} - lse_{max})) \\
&\quad\space= lse_{max} + \text{log1p}(\exp(lse_{min} - lse_{max})) \\
&\quad\space= lse_{max} + \text{softplus}(lse_{min} - lse_{max}) \\
&\text{where} \space lse_{max} = \max{(lse_1, lse_2)}, \space lse_{min} = \min{(lse_1, lse_2)}
\end{align}
$$

* where the last three steps are designed to address the $\exp$ explosion problem by extracting the maximum values as the additive term to prevent the exponential term from being positive large, along with the help of $\text{log1p}$ or $\text{softplus}$ operation for numerical stability (*See the Pytorch Log1p / Softplus Functional in [References](#references)*). Therefore, for each online attention step, we just need to apply the local block of attention to get $O_{bq_i}^{(bkv_j)}$ along with the local statistics $lse^{(bkv_j)}_{bq_i}$, and then update the global statistics $lse$ to calibrate the global output $O$ for the rows indexing in the range $[bq_i\cdot bq, (bq_i + 1)\cdot bq]$, as the equations shown above.

* To make full use of the implemented `OfflineSlidingWindowAttn` module in [task1](./task1.md), the `OnlineSlidingWindowAttn` module just inherits the `OfflineSlidingWindowAttn` module, where the input arguments are different in several ways as follows:
    * To simplify the diversity of inputs, the `OnlineSlidingWindowAttn` module only accepts the block of $Q_{bq_i},K_{bkv_j},V_{bkv_j}$ in `AttnQKVLayout.BSHD` layout and `AttnQKVPackFormat.Q_K_V` packing format, thus no arguments are required for the QKV packing format and layout.
    * Since the `sofmax clipping` and `softmax dropout` should only be applied to the global softmax weights $A$, we disable these two stabilization strategies in the `OnlineSlidingWindowAttn` module.
    * To better prepare for the online attention forward pass during the initialization, we provide `block_size` and `seqlen` for $Q$ and $K,V$ respectively in the argument list of `__init__` method. Therefore, you can pre-calculate something such as the full attention mask in the `__init__` method.
    * Since the layout is fixed to `AttnQKVLayout.BSHD`, we don't need neither `cu_seqlens_q` nor `cu_seqlens_kv` anymore in the argument list of the forward method.
    * The `q,k,v` arguments for the forward method are only a single block of $Q_{bq_i},K_{bkv_j},V_{bkv_j}$, where the $bq_i$ and $bkv_j$ are given as arguments `block_idx_q` and `block_idx_kv` respectively.
    * The global output $O$ and the global statistics $lse$ (*each entry is either partially updated already or set to the initial value as `0` for `O` and `-∞` for `lse`*) are given as arguments `global_o` and `global_lse` respectively, and you should update them **in-place**, thus no return value is needed for the forward method.


#### Summary

In summary, you should implement this `OnlineSlidingWindowAttn` module, which takes a block of $Q_{bq_i},K_{bkv_j},V_{bkv_j}$ in `AttnQKVLayout.BSHD` layout and `AttnQKVPackFormat.Q_K_V` packing format given the block index $bq_i$ and $bkv_j$, applies the local `offline sliding window attention` operation on this block, gets the local output $O_{bq_i}^{(bkv_j)}$ along with the local statistics $lse^{(bkv_j)}_{bq_i}$, and then updates the given global output $O$ and the global statistics $lse$ accordingly in-place.


#### Notice

* First of all, we inherit the same notice mentioned in [task1](./task1.md).
* The `dtype` and `device` of `q,k,v,global_o` are ensured to be the same, while we keep the `dtype` of `global_lse` as `torch.float32` to maintain the high precision to reduce the accumulation error.
* When the `seqlen` can not be fully divided by the `block_size`, the last in-complete block will be **padded** at the end of the sequence-dim to match the corresponding `block_size`, where the padding entries are filled with zeros.
* The `block_idx_q` and `block_idx_kv` are ensured to be in their corresponding valid ranges.
* **Note that** any online attention step in the forward pass of `OnlineSlidingWindowAttn` module should be regarded as an inner iterative step for the corresponding offline attention, i.e. if we tranverse each $bq_i \in [0, \frac{sq}{bq}]$ and $bkv_j \in [0, \frac{skv}{bkv}]$ on this online attention module, the final updated output $O$ should be the same as the corresponding offline attention module, ignoring the accumulation error.


#### References

*Hints: Here're some references which may be helpful to your task, or just deepen / broaden your knowledge to attention layers particularly in transformer:*

**!! Remember: it is a fundemental and essential capability to search, read, think and learn from the paper, source code, and official documentation for your answer, try NOT to rely too much on some biased and superficial blogs, e.g. CSDN !!**


* [Online Softmax Paper](https://arxiv.org/pdf/2112.05682)
* [LSE Wiki](https://en.wikipedia.org/wiki/LogSumExp)
* [Pytorch LSE Functional](https://pytorch.org/docs/stable/generated/torch.logsumexp.html#torch-logsumexp)
* [Pytorch Log1p Functional](https://pytorch.org/docs/stable/generated/torch.log1p.html#torch.log1p)
* [Pytorch Softplus Functional](https://pytorch.org/docs/stable/generated/torch.nn.functional.softplus.html#torch.nn.functional.softplus)
* [Nvidia Methods of Improving LLM Training Stability](https://arxiv.org/pdf/2410.16682)
* [Llama Attention Layer](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L275)
* [Google MHA paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
* [Google MQA paper](https://arxiv.org/pdf/1911.02150)
* [Google GQA paper](https://arxiv.org/pdf/2305.13245)
* [Pytorch Repeat Interleave Functional](https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html#torch.repeat_interleave)
* [Transformer paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
* [Flash Attention 2 Paper](https://arxiv.org/pdf/2307.08691.pdf)
* [Flash Attention Interface](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py)
* [Pytorch SDPA Functional](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention)
* [Pytorch FlexAttention Functional](https://pytorch.org/docs/main/nn.attention.flex_attention.html#module-torch.nn.attention.flex_attention)