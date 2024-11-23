
### Task 2: Sparse MLP with LoRA Adapters (60 points)

#### TODO

You are required to implement a pytorch module named `SparseMLPWithLoRA` in `src/modeling/mlp.py`.


#### Explanation

* Building upon the `DenseMLPWithLoRA` module described in [task1](./task1.md), we continue to implement the `SparseMLPWithLoRA` module with the leading Mixture-of-Experts (`MoE`) architecture (*See the MoE paper in [References](#references) for more details*).
* First of all, the "dense" `MLP` module generally refers to the common one that up-projects the hidden states `X` from `h`-dim to a higher `ffh`-dim, and then down-projects back with a gating mechanism. 
* Then, resembling to the `multi-head` mechanism in the attention module, the "sparse" version of the `MLP` module is proposed to split the `ffh` dimension of the projection matrices into `ne` equal shards, and each shard of size `e=ffh//ne` corresponds to one "expert" `Ei`, for any `i` in range `[0,..,ne-1]` (*`ne` denotes for the number of experts*). 
* Therefore, instead of "dense" projection with the large `ffh`-dim latent space, each token in the hidden states `X` is only mapped to `k` specific experts by a routing mechanism, each of which takes over only a specific `e`-dim sub-space (*For this `SparseMLPWithLoRA` module, you can simply model each expert as a small `DenseMLPWithLoRA` module, where the `ffh_size` argument is set to `e`*). And the final output for each token is the **weighted sum** of the sub-outputs from its `k` experts. In this manner, we can both decrease the high computation cost and increase the diversity of the latent patterns by a ratio of `ne`, similar to the `multi-head` mechanism.
* However, here come two questions to consider:
    * ① How can we model the routing mechanism to pick `k` specific experts for each token ?
    * ② How can we determine the weights `w` to weighted sum the sub-outputs from `k` experts for each token ?
* There are multiple solutions for the questions above. And as for this task, we choose the ones below, following `Mixtral` (*See the Mixtral paper in [References](#references) for more details*):
    * ① As shown in the equations below, we introduce an extra linear gating layer `G` with the shape `[h, ne]`. And for each single token `t`, we project each its `h`-dim hidden states to a `ne`-dim logits, and apply `softmax` to the logits to form the `ne`-dim routing probability distribution `Pt`, where `Pt[i]` indicates the probability for this token to be routed to expert `Ei`. Then we just pick `k` experts with the highest `k` routing probabilities as a subset `It` to route for this token, where the `k` routing probabilities can also form a new `k`-dim unnormalized distribution `Q_t`.
    * ② As shown in the equations above for each single token `t`, we can just define the weights `Wt` by renomalizing the `Qt` to a new `k`-dim routing probability distribution `Pt'`. And the weight `Wt[i]` for the sub-output `Ot'[i]` applied to expert `Ei` can be set to `Pt'[i]`.

$$
\begin{aligned}
    & P_t = \text{softmax}(X_tG), \quad where\space G \in \mathbb{R}^{h\times ne}, \space\forall t \\
    & I_t = \text{arg-topk}(P_t), \quad Q_t = P_t[I_t], \quad W_t = P_t' = \frac{Q_t}{\text{sum}(Q_t)}, \space\forall t \\
    & O_t'[i] = E_i(X_t), \space\forall i \in I_t, \space\forall t 
\end{aligned}
$$

* Furthermore, to simulate the distributed enviroment as we do in modeling `ParallelVocabEmbedding` (*Assignment1 - Task2*), we add two similar arguments `rank` and `world_size` to the `SparseMLPWithLoRA` module, which indicates that you should **ONLY** instantiate `nle` local experts for this module, with the expert index ranging in `R = [rank*nle, (rank+1)*nle)`, where `nle = ne // world_size` denotes the number of local experts of each rank in certain process group. Thus the final output `O_t` for each token `t` from this module is only a partial sum if the intersection set `I_t'` of `I_t` and `R` is not empty, otherwise setting it to an all-zero vector (*The complete final output can be obtained by summing the partial outputs from all ranks, but we ignore this process in this task*), as shown in the following equation.

$$
    O_t = \begin{cases}
    \sum\limits_{i \in I_t'} W_t[i] O_t'[i], & I_t' \neq \emptyset \\
    \vec{\mathbf{0}}, & I_t' = \emptyset
    \end{cases}, \quad where\space I_t' = I_t \space\cap\space R, \space\forall t
$$

* Finally, to initialize the weights of the gating layer `G` as well as the local experts, you should also implement the `reset_parameters` method for the `SparseMLPWithLoRA` module. As for experts, you can simply call their own `reset_parameters` methods (*but to avoid identical initialization across different experts, you should assign the offset `i` to any base seed for the expert `i`*). As for `G`, here we choose to initialize its weight from a **normal distribution** with the `init_mean` and `init_std` arguments, whose randomness is controlled by the `init_base_seed` **WITHOUT** any seed offset.

#### Summary

In summary, you should implement this `SparseMLPWithLoRA` module, which firstly initializes all the learnable parameters including the ones of each local expert in this module's charge, and the ones of the gating layer `G`, then takes `X` as input, for each token `t`, computes the top-`k` experts subset `I_t` to route from the distribution `P_t`, **ONLY** applies the forward pass for the tokens in `I_t'` (*leaving the output for other uncharged tokens as all-zero "holes"*), and finally returns the output hidden states `O` with the same shape as `X`, in which the non-zero final output `Ot` for some token `t` is the weighted sum of the sub-outputs from the local experts it's been routed to, and the summing weights `Wt` are determined by the renormalized distribution `Pt'`.


#### Notice

* First of all, we inherit the same notice mentioned in [task1](./task1.md) for the `DenseMLPWithLoRA` sub-module to model each single local expert.
* The `dtype` and `device` in arguments are for all the learnable parameters, which may be different from the ones of `X`.
* The meta attributes of the returned `O` including `dtype` and `device` should be the same as `X`. And you don't have to keep the meta attributes as `X` for any sub-output from any local expert, just only considering the final output `O`.
* The weight of the gating layer `G` usually requires higher precision as `float32` since the following sensitive `softmax` operation. Thus the param dtype of `G` is fixed to `float32`, regardless of the `dtype` argument.
* The `ffh` is ensured to be divisible by `ne` in all test cases, but it's still a good habit to check the divisibility in the `__init__` method.
* The `reset_parameters` method should be automatically called once in the `__init__` method when you initialize the parameters.
* The seed offsets of the parameters and dropout rates for each local expert have a two-level hierarchy as mentioned above. The first level is the expert index `i` (*note it's NOT the local expert index ranging from [0,..,nle-1], but ranging from [0,...,ne-1] to provide each expert with its own unique set of base seeds*), and the second level in the range of `{1,2,3}` is as defined in [task1](./task1.md) for projection matrices and lora weights respectively. **NO** seed offset is for the gating layer `G`.
 

#### References

*Hints: Here're some references which may be helpful to your task, or just deepen / broaden your knowledge to sparse-moe mlp layers in deep learning:*

**!! Remember: it is a fundemental and essential capability to search, read, think and learn from the paper, source code, and official documentation for your answer, try NOT to rely too much on some biased and superficial blogs, e.g. CSDN !!**

* [MoE Paper](https://arxiv.org/abs/1701.06538)
* [Mixtral Paper](https://arxiv.org/abs/2401.04088)
* [Mixtral MoE MLP Module](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/mixtral/modeling_mixtral.py#L610)
* [Llama MLP Module](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L229)
* [ChatGLM MLP Module](https://huggingface.co/THUDM/chatglm3-6b/blob/main/modeling_chatglm.py#L459)
* [GLU Paper](https://arxiv.org/abs/1612.08083)
* [GLU Variants Paper](https://arxiv.org/abs/2002.05202)
* [PEFT Documentation](https://huggingface.co/docs/peft/index)
* [LoRA Paper](https://arxiv.org/abs/2106.09685)
* [PEFT LoRA-Linear Layer Implementation](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora/layer.py#L400)
* [Pytorch SiLU Functional](https://pytorch.org/docs/stable/generated/torch.nn.functional.silu.html)
* [Pytorch GELU Functional](https://pytorch.org/docs/stable/generated/torch.nn.functional.gelu.html)
* [Pytorch ReLU Functional](https://pytorch.org/docs/stable/generated/torch.nn.functional.relu.html)
* [Pytorch Sigmoid Functional](https://pytorch.org/docs/stable/generated/torch.nn.functional.sigmoid.html)
* [Pytorch Kaiming Normal Initialization](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.kaiming_normal_)
* [Pytorch Xavier Normal Initialization](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.xavier_normal_)