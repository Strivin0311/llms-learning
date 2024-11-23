### Task 1: Dense MLP with LoRA Adapters (40 points)

#### TODO

You are required to implement a pytorch module named `DenseMLPWithLoRA` in `src/modeling/mlp.py`.


#### Explanation

* The Multi-Layer Perceptron (`MLP`) module is a fundamental building block in deep learning, particularly effective for tasks involving complex patterns and non-linear relationships. And with no doubts, it is widely adopted in modern transformer-based LLMs as the core component alongside the attention module.
* Speaking of the `MLP` modules specific to the prevailing LLMs such as `Llama`, they basically follow the Gated Linear Units (`GLU`) style (*See the GLU paper in [References](#references) for more details*) as shown in the equation below.

$$
\text{MLP}(X) = (\phi(XW_{gate}) \odot XW_{up})W_{down}, \quad where\space W_{up}, W_{gate} \in \mathbb{R}^{h\times \text{ffh}}, W_{down} \in \mathbb{R}^{\text{ffh}\times h}
$$

* where `X` denotes the input hidden states with the shape `[batch_size, seq_len, hidden_size]`, short as `[b, s, h]`. And $W_{up}$ is the up-projection matrix to project `X` from `h`-dim to `ffh`-dim, while $W_{down}$ does the opposite to down project `X` from `ffh`-dim to `h`-dim. To introduce non-linear transformations, resembling traditional deep RNN architectures, `GLU` employs a gate projection $W_{gate}$ with an activation function $\phi(\cdot)$ to form the gate $\phi(XW_{gate})$, applying elements-wise product $\odot$ onto up-projected $XW_{up}$ to control the flow of information during the forward pass and mitigate the vanishing gradient problem during the backward pass.

* So the first step is to implement the `GLU`-style MLP module defined above, but with some details added as follows:
    * The choice of the activation function $\phi(\cdot)$ is configurable for the `DenseMLPWithLoRA` module, by passing an argument named `activation_type`, which should be an instance of the enum class `MLPActivationType`, defined in `src/modeling/mlp.py` already. And the mapping from the `activation_type` to the corresponding activation function follows the `GLU Variants Paper` provided in [References](#references), as well as some pytorch implementations.
    * Same as any other learnable modules, you should implement `reset_parameters` method for the `DenseMLPWithLoRA` module to initialize the three projection matrices from the **normal distribution**. But different from the previous norm layers and embedding layers, we often apply either `Xavier Initialization`  or `Kaiming Initialization` to the projection matrices instead (*See more details in [References](#references)*). For this particular module, if the `activation_type` is set to `MLPActivationType.SIGMOID` or `MLPActivationType.BILINEAR`, we choose to apply `Xavier Initialization`, otherwise apply `Kaiming Initialization` for the rest `ReLU`-family activation functions.
    * To prevent identical initialization across different projection matrices since only one `init_base_seed` argument is given, we assign an unique seed offset for each projection matrix. Specifically, the seeds for up projection, gate projection, down projection are `init_base_seed+1`, `init_base_seed+2`, `init_base_seed+3`, respectively.

* Furthermore, since the parameters in `MLP` modules usually account for over 90% of the total trainable parameters in the LLMs, it is quite inefficient to apply the full-linear-parameter supervised fine-tuning (`SFT`), especially when the gain of linear parameters (*denoted as $\Delta(W)$*) is highly sparse.
* To apply parameter-efficient fine-tuning of LLMs (*short as `PEFT`, see more details in [References](#references)*), "Low-Rank Adaptation" method (*short as `LoRA`, see more details in [References](#references)*) is proposed to address this issue and now becomes one of the most popular strategies in `PEFT`. It is based on the prior that $\Delta(W)$ is a low-rank and sparse matrix with low-rank factorization $\Delta(W) = \frac{\alpha}{r}A_r B_r$, where $A_r \in \mathbb{R}^{h\times \text{r}}$ and $B_r \in \mathbb{R}^{\text{r}\times h}$ are the pair of low-rank factorized projection matrices, and $\alpha$ is a configurable scaling factor to control the magnitude of $\Delta(W)$. 
* Then the forward pass of the `MLP` module with `LoRA` adapters can be decomposed as the equation below, where the standard `Dropout` layer with the dropout rate `p` is introduced to reinforce the sparcity of $\Delta(W)$. In this way, we only need to tune the learnable $A_r$ and $B_r$ and froze any other pretrained projection matrices during `SFT`.

$$
\text{MLP}_{\text{LORA}}(X) = \text{MLP}(X) + \text{Dropout}_p(X \Delta(W)) = \text{MLP}(X) + \text{Dropout}_p(\frac{\alpha}{r}X A_r B_r)
$$

* As for the initialization of $A_r$ and $B_r$, we follow **the same rule** as the basic projection matrices in `DenseMLPWithLoRA` module. However, there're two differences as follows:
    * $A_r$ and $B_r$ should be initialized from the **uniform distribution**, no matter applying either `Xavier Initialization` or `Kaiming Initialization`.
    * We additionally give a `lora_init_base_seed` argument for the parameter initialization of `LoRA`, and the seeds for `A_r` and `B_r` are `lora_init_base_seed+1` and `lora_init_base_seed+2`, respectively.
    * To keep the reproducibility of the forward pass, we also introduce a `lora_dropout_seed` argument to control the random behavior of the `Dropout` layer.

#### Summary

In summary, you should implement this `DenseMLPWithLoRA` module, which firstly initializes all the learnable parameters including the basic projection matrices (*controlled by `init_base_seed`*) and the ones for `LoRA` adapters (*configued by `lora_rank`, `lora_alpha`, `lora_dropout_rate`, `lora_dropout_seed` and `lora_init_base_seed`*) respectively, then takes `X` as input, applies the forward pass of the `GLU`-style `MLP` with `LoRA` adapters with the specific activation function (*chosen by `activation_type`*) described above, and finally returns the output hidden states `O` with the same shape as `X`.


#### Notice

* We omit the addable bias in any linear projection.
* The `dtype` and `device` in arguments are for the learnable parameters, which may be different from the ones of `X`.
* The meta attributes of the returned `O` including `dtype` and `device` should be the same as `X`.
* The `reset_parameters` method should be automatically called once in the `__init__` method when you initialize the parameters.
* We fix to use the `fan_in` mode and `relu` nonlinearity if applying `Kaiming Initialization`. But it is notable that the pytorch implementation of `Kaiming Initialization` follows the `nn.Linear` style to regard weights in a transposed manner by default (*i.e., initialize `W.T` instead of `W`, which is also noted by pytorch itself*), thus be careful when you implement the `reset_parameters` method.
* The common usage of `LoRA` is a little bit simplified in this task, which only applies `LoRA` once for the whole `MLP` module. However in practice, we had better design a `LoRA` adapter for each linear projection matrix in the `MLP` module respectively.
* The `lora_rank` argument is ensured to be within the valid range of $[0, \min(h, ffh)]$. If the `lora_rank` is set to `0`, you should **skip any process** w.r.t `LoRA`, i.e. `LoRA` adaption is optional for this `DenseMLPWithLoRA` module.
* The `lora_alpha` argument is a positive scaling factor. By default, its value is set to `None`, which indicates that `lora_alpha` should be set to `lora_rank`.
* You can of course refer to how `Llama`, `ChatGLM`, `PEFT` build their `MLP` modules (*optionally with LoRA*) in [References](#references). However, please note that the specific requirements outlined above differ a little bit from theirs.


#### References

*Hints: Here're some references which may be helpful to your task, or just deepen / broaden your knowledge to dense mlp layers, lora adapters and activation functions in deep learning:*

**!! Remember: it is a fundemental and essential capability to search, read, think and learn from the paper, source code, and official documentation for your answer, try NOT to rely too much on some biased and superficial blogs, e.g. CSDN !!**


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
