# assignment-1-modeling-norm+emb
Starting with this assignment, we will design several modeling tasks for you to get familiar with transformer's component modules gradually. 

As for this one, we will focus on modules in the transformer structure including norm layers and embedding layers.


## Tasks

### Task 1: Group RMS Normalization

Please read the description [here](./tasks/task1.md).

### Task 2: Parallel Vocab Embedding

Please read the description [here](./tasks/task2.md).

### Task 3: NTK-aware RoPE

Please read the description [here](./tasks/task3.md).


## Environment

* You should have python 3.10+ installed on your machine.
* (*Optional*) You had better have Nvidia GPU(s) with CUDA12.0+ installed on your machine, otherwise some features may not work properly (*We will do our best to ensure that the difference in hardware does not affect your score.*).
* You are supposed to install all the necessary dependencies with the following command, **which may vary a little among different assignments**.
    ```python
    pip install -r requirements.txt
    ```
* (*Optional*) You are strongly recommended to use a docker image from [Nvidia Pytorch Release](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html) like [23.10](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-10.html#rel-23-10) or some newer version as your basic environment in case of denpendency conflicts.