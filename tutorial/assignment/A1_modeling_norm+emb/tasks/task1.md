
### Task 1: Group RMS Normalization (20 points)

#### TODO

You are required to implement a pytorch module named `GroupRMSNorm` in `src/modeling/norm.py`.


#### Explanation

* Root Mean Square Layer Normalization (RMS Norm) is one of the most widely-used normalization modules in DL, especially in NLP and LLMs (*See the paper in [References](#references)*).
* This module takes certain hidden states tensor with the shape `[batch_size, seq_len, hidden_size]` as input (*denoted as `X`, with the shape: `[b, s, h]`*), and apply root-mean-square normalization with learnable scaling transformation along the hidden dimension (*denoted as `Y`, with the same shape `[b, s, h]`*), as shown in the equation below.

$$
Y = \frac{X}{\sqrt{\text{RMS}[X]^2 + \epsilon}} \cdot \gamma
$$

* where $\text{RMS}[X]$ is the root-mean-square of `X`, $\epsilon$ is a small constant to avoid division by zero (*denoted as `eps`*), and $\gamma$ is the learnable per-channel scaling parameter applied along the hidden dimension.

* To generalize it, here we implement a simple variant of RMS Norm, called "Group RMSNorm". Given the `group size` denoted as `gz`, it evenly splits the hidden dimension of `X` into groups (*denoted as `Xg`*), and apply the same root-mean-square normalization with learnable scaling transformation as RMS Norm, just **on each `i`-th group individually**, as shown in the equation below.

$$
Y_{g_i} = \frac{X_{g_i}}{\sqrt{\text{RMS}[X_{g_i}]^2 + \epsilon}} \cdot \gamma_{g_i}
$$

* By the way, you should also implement an parameter initialization member method called `reset_parameters` for this `GroupRMSNorm` module class (*which is a standard method to initialize parameters for a learnable pytorch module*), which initializes the learnable scaling parameter $\gamma$ from a **uniform distribution** given the range tuple (*denoted as `init_range`*) like `(-1, 1)`, and random seed (*denoted as `init_seed`*) like `42`.

#### Summary

In summary, you should implement this `GroupRMSNorm` module, which firstly initializes the learnable scaling parameter $\gamma$ from a uniform distribution controlled by `init_range` and `init_seed`, and then takes `X` and `gz` as input, and apply normalization described above to return `Y` (*not `Yg`*).


#### Notice

* The `dtype` and `device` in arguments are for the learnable scaling parameter $\gamma$, which may be different from the ones of `X`.
* The meta attributes of the returned `Y` including `dtype` and `device` should be the same as `X`.
* Since root-mean-square normalization involves division operations, it requires higher precision as `float32`.
* The `h` is ensured to be divisible by `gz` in all test cases, but it's still a good habit to check the divisibility in the `__init__` method.
* The `reset_parameters` method should be automatically called once in the `__init__` method when you initialize the parameters.
* You can of course refer to how `Llama` and `ChatGLM` build their RMS Norm modules in [References](#references), but be careful about the specific requirments above, which differ a little from the ones of neither `Llama` nor `ChatGLM`.


#### References

*Hints: Here're some references which may be helpful to your task, or just deepen / broaden your knowledge to normalization layers in deep learning:*

**!! Remember: it is a fundemental and essential capability to search, read, think and learn from the paper, source code, and official documentation for your answer, try NOT to rely too much on some biased and superficial blogs, e.g. CSDN !!**

* [RMSNorm Paper](https://arxiv.org/abs/1910.07467)
* [Pytorch RMSNorm Module](https://pytorch.org/docs/stable/generated/torch.nn.RMSNorm.html#rmsnorm)
* [Llama RMSNorm Module](https://github.com/huggingface/transformers/blob/v4.46.3/src/transformers/models/llama/modeling_llama.py#L60)
* [ChatGLM RMSNorm Module](https://huggingface.co/THUDM/chatglm3-6b/blob/main/modeling_chatglm.py#L181)
* [Pytorch LayerNorm Module](https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html#torch.nn.LayerNorm)
* [Pytorch BatchNorm Module](https://pytorch.org/docs/stable/generated/torch.nn.BatchNorm1d.html#torch.nn.BatchNorm1d)
* [Pytorch GroupNorm Module](https://pytorch.org/docs/stable/generated/torch.nn.GroupNorm.html#torch.nn.GroupNorm)
* [Pytorch Uniform Initialization](https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.uniform_)
