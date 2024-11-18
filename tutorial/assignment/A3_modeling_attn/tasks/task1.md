### Task 1: Offline Sliding-Window Attention (60 points)

#### TODO

You are required to implement a pytorch module named `OfflineSlidingWindowAttn` in `src/modeling/attention.py`.


#### Explanation

* The multi-head `Attention` module is an essential building block in Transformer (*See the Transformer paper in [References](#references)*) . It takes three tensors as inputs: query tensor with the shape `[batch_size, seq_len_q, num_head_q, head_dim]` (denoted as $Q$ with the shape `[b, sq, hq, hd]`), key tensor and value tensor with the same shape `[batch_size, seq_len_kv, num_head_kv, head_dim]` (denoted as $K$, $V$ with the shape `[b, skv, hkv, hd]` respectively). 
* Each row tensor $q_i$ in $Q$ can be represented as an embedded latent "query message" for the $i$-th token, inquiring for some knowledge embedded in a "knowledge base" $V$, where each row tensor $v_j$ can be viewed as an embedded latent "knowledge archive" for the $j$-th token.
* To aggregate all important knowledge in $V$ and ignore other irrelevant ones,  each $v_j$ corresponds to an embedded latent "key word" $k_j$, where the dot-product scalar $q_i^{\text T}k_j$ of any $q_i$ with this $k_j$ can be seen as the "similarity score" between the query message $q_i$ with this knowledge archive $v_j$. 
* Thus the aggregate knowledge $o_i$ for each query $q_i$ can be represented as a weighted-sum of all $v_j$ in $V$ as $o_i := \sum\limits_j a^{(i)}_jv_j$, where the weight vector $a^{(i)}$ is comprised of all "normalized" dot-product similarity scalars of $q_i$ with each $k_j$, as mentioned above.
* As for the "normalization" of weights, the most common choice is to apply `softmax` operation, which is known as the "soft" `maximalization` operation, in order to only "pay attention to" the knowledge that really matters with the highest similarity scores.
* Therefore, the whole `attention` operation can be simply written as (*for each batch and each head*):

$$
\begin{align}
&\text{Attention}(Q, K, V) = AV, \\
&\text{where} \space A = \text{softmax}_{row-wise}(P) \in \mathbb{R}^{sq\times skv}, \\
&\space P = QK^{\text T} + M\in \mathbb{R}^{sq\times skv}
\end{align}
$$

* where $M$ denotes the binary attention mask where each entry is valued in $\{-\infty, 0\}$, to either mask out the irrelevant pairs of $(q_i, k_j)$ with the value of $-\infty$, or keep the relevant pairs with the value of $0$. For example, to apply `causal language modeling`, any token can only attend to the previous tokens and itself, i.e. $q_i$ can only attend to $k_j$ where $j \le i$ at most.

* For this `OfflineSlidingWindowAttn`, we need to implement the $M$ in the `sliding window` style, i.e. $q_i$ can only attend to $k_j$ where $j \in [i-w, i+w]$, where $w$ denotes the window size, and in addition to the `causal` style, $q_i$ is only allowed to attend to $k_j$ where $j \in [i-w, i]$.

* Moreover, since `softmax` operation is known for its sensitivity, we had better apply some strategies to stablize it. The most common one is to apply `softmax scale` to $P$ as $scale \cdot P$, where $scale$ is often set to $\frac{1}{\sqrt{hd}}$, to avoid the surging values with the dimension scales up. Most recently, Nvidia introduces some other tricks to improve the stability of `softmax` operation during training (*See the Nvidia paper in [References](#references)*), we will also adopt some of them into the `OfflineSlidingWindowAttn` module as follows.
    * 1. `softmax temperature`: to control the sharpness of the softmax distribution, we can apply `softmax temperature` to $P$ as $\frac{P}{temp}$, where $temp$ is ranged in $(0, +\infty)$. When setting $temp$ to $1.0$, the distribution is of-course the original one, and the closer $temp$ is to $0.0$ ($+\infty$), the sharper (smoother) the distribution becomes.
    * 2. `softmax capping`: to adaptively control the magnitude of $P$ except for `softmax temperature`, we can apply `softmax capping` to $P$ as $cap\cdot \text{tanh}(\frac{P}{cap})$, where $cap$ is often a large positive number. Since it can be seen as an adaptive version of `softmax temperature`, we will **only apply either one of them** in one forward pass.
    * 3. `softmax clipping`: to prevent the outliers in $A$ from growing, we can apply `softmac clipping` to $A$ as $\text{clip}((r-l)\cdot A + l, 0, 1)$, where the range $[l,r]$ is super-range over $[0,1]$, i.e. $l \le 0$ and $r \ge 1$, to affine the value range of $A$ from $[0,1]$ to $[l,r]$, and then clip back to $[0,1]$, to cut-off the outliers.
    * 4. `softmax dropout`: to improve the robustness of $A$, we can apply `softmax dropout` to $A$ as $\text{dropout}_p(A)$ with the dropout rate $p \in [0,1]$.
    * 5. `QK layer normalization`: to further address the large values in $P$ which lead to attention weights degeneration (i.e. $A$ almost becomes one-hot), we optionally pre-apply `layer normalization` to $Q$ and $K$ respectively. But for this assignment, we change the `layer normalization` to `group rms normalization`, to fully make use of the `GroupRMSNorm` module we implemented in the previous assignment.

* Therefore, the whole `offline sliding window attetion` operation can be written as (*for each batch and each head*):


$$
\begin{align}
&\text{OfflineSlidingWindowAttention}(Q, K, V) = \widehat AV, \\
&\text{where} \space \widehat A = \text{dropout}\space_p(\text{clip}((r-l)\tilde A + l, 0, 1)), \\
&\space \tilde A = \text{softmax}\space_{row-wise}(\tilde P), \\
&\space \tilde P = \begin{cases}
\cfrac{scale\cdot \tilde Q \tilde K^{\text T}}{temp} + M_{sw} \space(+ M_{causal}), & \text{softmax temperature} \\
cap\cdot \text{tanh}(\cfrac{scale\cdot \tilde Q\tilde K^{\text T}}{cap}) + M_{sw} \space(+ M_{causal}), & \text{softmax capping} \\
\end{cases}, \\
&\text{where}\space \tilde Q = \text{GroupRMSNorm}(Q), \space \tilde K = \text{GroupRMSNorm}(K) \\
\end{align}
$$


* In the meanwhile, to make the `OfflineSlidingWindowAttn` module more flexible for different inputs:
    * 1. We design an enum class named `AttnQKVPackFormat` in `src/modeling/attention.py`, which defines how $Q,K,V$ are packed as inputs:
        * `AttnQKVPackFormat.Q_K_V`: the most common format where $Q,K,V$ are three seperate tensors without packing.
        * `AttnQKVPackFormat.Q_KV`: $K,V$ are packed together along the "num_heads" dimension, while $Q$ is an individual tensor.
        * `AttnQKVPackFormat.QKV`: $Q,K,V$ are all packed together along the "num_heads" dimension (*in such case, the other dimensions of Q and KV have to be the same*).
    * 2. We desing another enum class named `AttnQKVLayout` in `src/modeling/attention.py`, which defines the shape layout of $Q,K,V$:
        * `AttnQKVLayout.BSHD`: the most common layout where $Q,K,V$ follow the shape like "bshd".
        * `AttnQKVLayout.SBHD`: the more friendly layout than "bshd" in the distributed environment where $Q,K,V$ follow the shape like "sbhd".
        * `AttnQKVLayout.THD`: the most general layout where $Q,K,V$ follow the shape like "thd" (*a.k.a. "varlen" layout*), i.e. there is no explicit "batch"-dim, and all individual sequences with variable length are concatenated along the "sequence"-dim. Thus in this case, there will be two additional inputs to help indexing the inner batch of sequences: `cu_seqlens_q` and `cu_seqlens_kv`, each of which is an int32 tensor with shape `[batch_size+1,]`, where `[cu_seqlens[i], cu_seqlens[i+1])` denotes the `[start, end)`-like index range of the $i$-th sequence in $Q$ and $K,V$ respectively (*See the Flash Attention Interface in [References](#references) for more examples*).


#### Summary

In summary, you should implement this `OfflineSlidingWindowAttn` module, which takes $Q,K,V$ in different packing formats and different layouts as inputs (along the `cu_seqlens_q` and `cu_seqlens_kv` if the layout is `AttnQKVLayout.THD`), applies the `offline sliding window attention` operation described above, and returns the output tensor $O$ in the same layout as $Q$.

#### Notice

* The `dtype` and `device` in arguments are for the learnable parameters in `GroupRMSNorm`, which may be different from the ones of $Q,K,V$.
* The meta attributes of the returned $O$ including `dtype`, `device` and `layout` should be the same as $Q$.
* Only if the argument `softmax_cap` is set to `None`, can we apply the `softmax temperature` strategy with the argument `softmax_temp`.
* All the arguments are ensured to be in their valid ranges.
* The `GroupRMSNorm` of $Q,K$ are individual sub-layers of `OfflineSlidingWindowAttn`, since `GroupRMSNorm` only accept the 3D tensor with the shape `[batch_size, seq_len, hidden_size]`, and the `hidden_size = num_heads * head_dim` may vary between $Q$ and $K$. Moreover, we ensure the `head_dim` can be divided by `group_size`, i.e. no group will cross two different heads along the hidden dimension.
* When the "num_heads"-dim are different between $Q$ and $K,V$ (*i.e. in the `MQA` style or `GQA` style, see the papers in [References](#references) for more details*), we follow the same **"kv-heads repeating" strategy** to make $Q$ and $K,V$ match in the number of heads (*See the Llama Attention Layer and the Pytorch Repeat Interleave Functional in [References](#references) for more details*).
* When the "sequence"-dim are different between $Q$ and $K,V$ (*e.g. in the cross-attention or the autoregressive decoding phase*), the attention mask $M$ is not a square matrix but a rectangle matrix with the shape `[sq, skv]`, also seen as a "slide" of the latent full square matrix with the shape `[max(sq,skv), max(sq,skv)]`. Hence there comes a question: which rectangle slide should we choose from the full matrix? For this `OfflineSlidingWindowAttn` module, we'd like to choose the one which aligns the **bottom right part** of the full square attention matrix following the flash-attention's settings (*See the Flash Attention Interface in [References](#references) for more examples*).


#### References

*Hints: Here're some references which may be helpful to your task, or just deepen / broaden your knowledge to attention layers particularly in transformer:*

**!! Remember: it is a fundemental and essential capability to search, read, think and learn from the paper, source code, and official documentation for your answer, try NOT to rely too much on some biased and superficial blogs, e.g. CSDN !!**


* [Nvidia Methods of Improving LLM Training Stability](https://arxiv.org/pdf/2410.16682)
* [Llama Attention Layer](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L277)
* [Google MHA paper](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
* [Google MQA paper](https://arxiv.org/pdf/1911.02150)
* [Google GQA paper](https://arxiv.org/pdf/2305.13245)
* [Pytorch Repeat Interleave Functional](https://pytorch.org/docs/stable/generated/torch.repeat_interleave.html#torch.repeat_interleave)
* [Transformer paper](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
* [Flash Attention 2 Paper](https://arxiv.org/pdf/2307.08691.pdf)
* [Flash Attention Interface](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_interface.py)
* [Pytorch SDPA Functional](https://pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention)
* [Pytorch FlexAttention Functional](https://pytorch.org/docs/main/nn.attention.flex_attention.html#module-torch.nn.attention.flex_attention)