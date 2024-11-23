
### Task 3: NTK-aware RoPE (50 points)

#### TODO

You are required to implement a pytorch module called `NTKAwareRoPE` in `src/modeling/pos_emb.py`, and a pytorch function called `apply_rotary_pos_emb` in `src/functional.py`.


#### Explanation

* Transformers process input tokens in parallel as a bag-of-words and lack an inherent sense of sequence order. To preserve the sequential information, the vanilla Transformer presents a novel Sinusoidal PE (`SinPE`), as shown in the equation below (*See the first paper/blog in [References](#references) for more details*).

$$
\mathrm{SinPE}(n) :=
    \left[\begin{matrix}
        \sin(n\theta^0) \\
        \cos(n\theta^0) \\
        \sin(n\theta^1) \\
        \cos(n\theta^1) \\
        \vdots\\
        \sin(n\theta^{\frac{d}{2}-1})\\
        \cos(n\theta^{\frac{d}{2}-1})\\
    \end{matrix}\right], 
    \quad where\quad  \theta := \beta^{-1}, \space  \beta := base^{\frac{2}{d}}, \space n\in\{0,1,\cdots, L-1\}
$$

* Where `L` denotes the sequence length, `d` denotes the hidden dimension, `base` is a large integer manually set as `10000` (*according to the original paper without further explanation*), and `𝛽` represents the power basis of the wavelength or period of the trigonometric basis functions, which increases as a geometric series $\{\beta^i\}_{i=0}^{d/2}$ with the dimension `𝑖` goes deeper.

* The Rotary Position Embedding (`RoPE`) provides a more stable scheme to handle longer sequences. It captures relative positional patterns with absolute position awareness, thus widely used in state-of-the-art open-source LLMs like `LLama` and `ChatGLM`. And gradually, it replaces original `SinPE`, learnable PE, relative PE, among others in the transformer and becomes the standard PE choice in current LLMs due to its stability and flexibility.

* More specifically, it applies a rotation operation on a complex field instead of an addition to the hidden states based on absolute positions, where it shares the same basis function as SinPE, as shown in the following equation (*See the second paper/blog in [References](#references) for more details*).

$$
\mathrm{RoPE}(n) := \left[
    \begin{matrix}
        R_n^{(0)}\\
        \space  & R_n^{(1)}\\
        \space  & \space  & \ddots\\
        \space  & \space  & \space  & R_n^{(\frac{d}{2}-1)}\\
    \end{matrix}\right], \quad  where\quad  R_n^{(i)} := \left[\begin{matrix}
        \cos(n\theta^i) & -\sin(n\theta^i)\\
        \sin(n\theta^i) & \cos(n\theta^i)\\
    \end{matrix}\right]
$$

* However, even though `RoPE` offers advantages such as relative distance decay, training stability, it still lacks satisfactory **length extrapolation capabilities** on sequence length, i.e. "Train Short and Test Long" (*See the `Length Extrapolation` papers in [References](#references) for more details*). Therefore, several research works have aimed to extend `RoPE` to generalize its valid inference sequence length beyond the one used in training much farther.

* Among them, `NTK-aware RoPE` combines high-frequency extrapolation and low-frequency interpolation together. It scales `𝛽` using coefficient `𝑐𝜅` to achieve equivalence during interpolation by a ratio of `𝜅` for the lowest frequency term while maintaining scale for terms with higher frequency, as shown in the equation below. Surprisingly, this nonlinear scaling can be directly applied to LLMs pretrained with RoPE, like `Llama` (*See the `Llama RoPE` source codes in [References](#references) for more details*), without further finetuning to extend the context length boundary, adopted in `CodeLlama`.

$$
\widetilde\beta := c_{\kappa}\cdot\beta, \quad s.t.\quad \cfrac{n}{\widetilde\beta^{d/2-1}} = \cfrac{n/\kappa}{\beta^{d/2-1}} \Rightarrow c_{\kappa} = \kappa^{2/(d-2)}
$$

* In this task, you are required to implement the `NTKAwareRoPE` module just as `Llama` does, however, there're several differences described as follows:
    * 1. The standard `RoPE` module's forward pass only returns the cos/sin basis tensors, each with the shape `[seq_len, head_dim]` (*denoted as `(C, S)` with the shape `[s, hd]`*), and applies the rotary embedding in another individual function usually named `apply_rotary_pos_emb`. 
    * 2. We basically follows the style that you are supposed to implement this `apply_rotary_pos_emb` function in the `src/functional.py`, but it is also imported in `src/modeling/pos_emb.py` and should be called in `NTKAwareRoPE`'s `forward` method to **NOT** just return `(C, S)`, but also apply the rotary embedding operation to the given input tensor with the shape `[batch_size, seq_len, num_heads, head_dim]` (*denoted as `X` with the shape `[b, s, nh, hd]`*) and **ONLY** return the embedded output tensor (*denoted as `E` with the same shape as `X`*).
    * 3. Another common issue is that when we firstly initialize the `NTKAwareRoPE` with a given maximum sequence length used in training (*denoted as `ms`*), and a scaling ratio (*denoted as `k`*), we can already prepare the `(C, S)` with the shape `[es, hd]` in advance, where the maximum extended sequence length we can support is fixed to `es = ms * k`. Therefore, once a `X_` with its sequence length `s_` larger than `es`, is fed into the forward method, we have to recompute a new pair of `(C_, S_)` to satisfy this "outier".
    * 4. But here come two questions: 
        * ① To recompute the new pair of `(C_, S_)`, how should we decide the new scaling ratio `k_` for `X_` ?
        * ② Should we just only give the temporary `(C_, S_)` each time when an "outlier" comes and still keep the original `k` and `(C, S)` for the normal ones, or, should we update `k` and `(C, S)` with the new `k_` and `(C_, S_)` respectively each time ?
    * 5. There is no standard answer for the questions above yet. In this task, we use the strategies as follows:
        * ① When a new `s_ > es` comes, we choose the **minimum** `k_` that satisfies the new `es_ = ms * k_ >= s_`, and `k_` is an even integer.
        * ② We add a bool argument `dynamic` when initializing the `NTKAwareRoPE` module. And if `dynamic` is `True`, then each time, we update `k <- k_` and `(C, S) <- (C_, S_)` each time, otherwise, we just give the temporary `(C_, S_)` for the specific "outlier" and keep `k` and `(C, S)` constant.


#### Summary

In summary, you should implement a pytorch module named `NTKAwareRoPE`, which firstly initializes the origianl `(C, S)` given `hd`, `ms`, `base`, `k`, then takes `X` with the shape `[b, s, nh, hd]` as input, either retrieves the cached `(C, S)` when `s <= es` or recomputes the new `(C_, S_)` with the new `k_` when `s > es`. And in the latter case, it further updates the internal `k` and `(C, S)` if the given argument `dynamic` is set to `True`. Finally, it applies the rotary embedding operation to `X` with its corresponding `(C, S)` by calling the `apply_rotary_pos_emb` function which you are supposed to implement and returns `E`.

#### Notice

* The `dtype` and `device` in arguments are for the `(C, S)`. And usually we require higher precision for positional embedding, thus we will fix `dtype` to `float32` in all test cases and you had better use `float32` through every step of the computations.
* The returned `E` should shares the same dtype and device with `X`.
* In practical, the `(C, S)` should be regarded as module states to not only swtich its device with the module (*e.g., `module.to(device)`*), but also be discarded when saving the state dict as it is easily to be reconstructed. Therefore, you should **NOT** assign `(C, S)` to `self` directly as ordinary python attributes, but register it as the pytorch non-persisitent buffer (*See `Pytorch Module Register` in [References](#references) for more details*).
* You can of course refer to how `Llama` and `ChatGLM` build their RoPE modules in [References](#references), but be careful about the specific requirments above, which differ from the ones of neither `Llama` nor `ChatGLM`.

#### References

*Hints: Here're some references which may be helpful to your task, or just deepen / broaden your knowledge to position embedding layers in deep learning:*

**!! Remember: it is a fundemental and essential capability to search, read, think and learn from the paper, source code, and official documentation for your answer, try NOT to rely too much on some biased and superficial blogs, e.g. CSDN !!**

* `SinPE`: [paper](https://arxiv.org/abs/1706.03762) | [blog](https://spaces.ac.cn/archives/8231)
* `RoPE`: [paper](https://arxiv.org/abs/2104.09864) | [blog](https://spaces.ac.cn/archives/8265)
* `Length Extrapolation`: [Alibi](https://arxiv.org/abs/2108.12409) | [PI](https://arxiv.org/abs/2306.15595)
* `NTK-aware RoPE`: [blog](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) | [paper](https://arxiv.org/abs/2309.00071) | [survey](https://arxiv.org/abs/2311.12351)
* `Llama RoPE`:  [module](https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/models/llama/modeling_llama.py#L178) | [function](https://github.com/huggingface/transformers/blob/v4.36.2/src/transformers/models/llama/modeling_llama.py#L211)
* `ChatGLM RoPE`: [module](https://huggingface.co/THUDM/chatglm3-6b/blob/main/modeling_chatglm.py#L121) | [function](https://huggingface.co/THUDM/chatglm3-6b/blob/main/modeling_chatglm.py#L121)
* `Pytorch Module Register`: [buffer](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_buffer) | [parameter](https://pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_parameter)