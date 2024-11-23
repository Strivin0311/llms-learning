
### Task 1: MatMul with Importance (100 points)

#### TODO

You are required to implement a python function named `matmul_with_importance` in `src/functional.py` with pytorch.

#### Explanation

* According to its docstring, this function is to apply a special variant of matrix multiplication operation (denoted as `matmul`) of two tensors, where:
    * the input tensor is a 3D tensor with the shape `[batch_size, seq_len, hidden_size]`, which represents `batch_size` of sequences, and each sequence has `seq_len` elements, where each element is a row-vector with dimension `hidden_size`. We denote the input tensor as `A1` with the shape `[b, s, h]`.
    * the weight tensor is a 2D tensor with the shape `[hidden_size, embed_size]`, which represents a projection matrix that projects any row vector from `hidden_size`-dim to `embed_size`-dim. We denote the weight tensor as `W1` with the shape `[h, e]`.
* The naive `matmul` is just to apply `O1[i] = A1[i] @ W1` for each `i`-th sequence in the batch to get output tensor denoted as `O1` with the shape `[b, s, e]`.
* The multi-head variant of `matmul` involves splitting the `h` dimension of `A1` and `W1` into `num_heads` shards equally (*denoted as `nh`, provided as an argument*) and performing `matmul` on each pair of `A1` shard and `W1` shard individually. This transforms the input tensor into a 4D tensor, denoted as `A2` with the shape `[b, s, nh, hd]`, and accordingly transforms the weight tensor into a 3D tensor denoted as `W2` with the shape `[nh, hd, e]`. As a result, the output tensor becomes a 4D tensor as well, denoted as `O2` with the shape `[b, s, nh, e]`.
* Building on the multi-head version of `matmul`, we introduce an importance probability tensor, denoted as `P`, with the shape `[b, s]`. Each element in `P` represents the probability of how important the corresponding element in `A1` is relative to other elements within the same sequence.
* As a result, we aim to apply matmul only to the "important" elements in each sequence. The projected vectors of these important elements, totaling `total_important_seq_len` (*denoted as `t`*), are then gathered into an output tensor, denoted as `O3` with the shape `[t, nh, e]`.
* To precisely define what is considered "important", we provide two optional arguments:
    * `top_p`: A float in the range `[0., 1.]`. Only elements with a probability **equal to or higher** than `top_p` are considered "important". The default value is `1.0`.
    * `top_k`: An integer in the range `[1, ..., seq_len]`. For each sequence in the batch, only the elements with the `top_k` highest probabilities are considered "important". If `top_k` is not provided (default is `None`), it is treated as `top_k = seq_len`.
* Additionally, if the optional gradient of the output tensor (*`grad_output`, denoted as `dO3` with the same shape as `O3`*) is provided, we should also compute the gradients for the input tensor (*`grad_input`, denoted as `dA1` with the same shape as `A1`*) and the weight tensor (*`grad_weight`, denoted as `dW1` with the same shape as `W1`*). If `dO3` is not given, we return `None` for both `dA1` and `dW1`.

#### Summary

In summary, the core of the `matmul_with_importance` function is to compute and return a tuple of three (optional) tensors: either `(O3, dA1, dW1)` or `(O3, None, None)`, given as input two tensors `A1` and `W1` as `matmul` operators, an importance probability tensor `P` with `top_p` and `top_k` to control "importance", an optional gradient tensor `dO3`, and an integer `num_heads` to split the `hidden_size`.

#### Notice

* All given tensors are randomly initialized from the standard normal distribution `N(0, 1)` on the same device (either `cpu` or `cuda`), with the same data type (`float32`, `float16`, or `bfloat16`), and do **NOT** require gradients in any test cases.
* `top_p` and `top_k` are guaranteed to have valid values within their respective ranges in all test cases.
* `hidden_size` is guaranteed to be divisible by `num_heads` in all test cases.
* If `grad_output` is not provided, avoid computing gradients to improve efficiency and save memory.
* If `grad_output` is provided, you can compute gradients using pytorch’s autograd mechanism, but be cautious of potential **side effects**, which may be checked in the test cases.
* There are so many ways to perform the `matmul` operation in pytorch, including `@`, `torch.matmul`, `torch.mm`, `torch.bmm`, and `torch.einsum`, among others. It is recommended that you experiment with different methods to implement the task and explore the differences between them.


#### References

*Hints: Here're some references which may be helpful to your task, or just deepen / broaden your knowledge to pytorch:*

**!! Remember: it is a fundemental and essential capability to search, read, think and learn from the official English documentation for your answer, try NOT to rely too much on some biased and superficial blogs, e.g. CSDN !!**

* [Pytorch Documentation](https://pytorch.org/docs/stable/index.html)
* [Pytorch Autograd Mechanism](https://pytorch.org/docs/stable/autograd.html#module-torch.autograd)
* [PyTorch Internals](http://blog.ezyang.com/2019/05/pytorch-internals/)