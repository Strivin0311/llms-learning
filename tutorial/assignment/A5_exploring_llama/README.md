# assignment-5-exploring-llama 🦙 (final plus bonus🔥)
Congratulations! 

You've journeyed from starting with pytorch to constructing a complete transformer, equipping you the capability to master most of the modern LLMs. 

So in this assignment, we dive into Llama3.2, a SOTA lightweight dense LLM, exploring its end-to-end pipeline, including modeling, inference, and training.

🔥 What'more, as the final assignment, for each basic task, we provide some additional hight-relative tasks but less guided and more challenging, giving you enough room to improvise and explore.


## Tasks (plus bonus🔥)

### Task 1: Llama Model (plus bonus🔥)

Please read the description [here](./tasks/task1.md).

### Task 2: Inference Agent (plus bonus🔥)

Please read the description [here](./tasks/task2.md).

### Task 3: LoRA Trainer (plus bonus🔥)

Please read the description [here](./tasks/task3.md).


## Environment

* You should have python 3.10+ installed on your machine.
* (*Optional*) You had better have Nvidia GPU(s) with CUDA12.0+ installed on your machine, otherwise some features may not work properly (*We will do our best to ensure that the difference in hardware does not affect your score.*).
* You are supposed to install all the necessary dependencies with the following command, **which may vary a little among different assignments**.
    ```python
    pip install -r requirements.txt
    ```
* (*Optional*) You are strongly recommended to use a docker image from [Nvidia Pytorch Release](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html) like [23.10](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-10.html#rel-23-10) or some newer version as your basic environment in case of denpendency conflicts.


## Notice 🛎️❗

* Since the numerical errors might accumulate too much to be close to the reference implementation with the model goes deeper, as for the final assignment, we do **NOT** continue to adopt pytest-based test cases requiring exact closeness, neither to help you debug nor to evaluate your implementation.
* Instead, we will provide each task an ipython notebook named `test_toy_task{i}.ipynb` with the exported python script named `test_toy_task{i}.py` for the `i`-th task, in which we write down a toy tutorial to go through main functionalities that you're required to implement. In this way, you can directly run the notebook to debug and evaluate your own implementation, focusing **NOT** on precision, but on reasonableness.