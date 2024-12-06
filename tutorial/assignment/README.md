# The Assignments for the LLMs Intro Course

## Assignments

### Assignment 0: On-Boarding

README: [here](./A0_onboarding/README.md)

This assignment is designed to help you get familiar with the programming environment, submission process and basic pytorch functionalities. 

By completing it, you'll ensure that your development setup is working properly, understand how to submit your work for future assignments, and strengthen your pytorch coding skills.


### Assignment 1: Modeling Norm and Embedding

README: [here](./A1_modeling_norm+emb/README.md)

Starting with this assignment, we will design several modeling tasks for you to get familiar with transformer's component modules gradually. 

As for this one, we will focus on modules in the transformer structure including norm layers and embedding layers.

### Assignment 2: Modeling MLP

README: [here](./A2_modeling_mlp/README.md)

For this assignment, we are continuing to design modeling tasks to help you gain a deeper understanding of transformer's component modules. 

Specifically, we will focus on one of the pivotal layers that form the backbone of the transformer structure: the mlp layer.

### Assignment 3: Modeling Attention

README: [here](./A3_modeling_attn/README.md)

For this assignment, we are continuing to design modeling tasks to help you gain a deeper understanding of transformer's component modules. 

Specifically, we will focus on one of the pivotal layers that form the backbone of the transformer structure: the attention layer.

### Assignment 4: Modeling Transformer

README: [here](./A4_modeling_transformer/README.md)

Finally! 

For this assignment, we are going to bring all the modules we've built in the previous assignments together, and construct a complete decoder-only transformer.

### Assignment 5: Exploring Llama

README: [here](./A5_exploring_llama/README.md)

Congratulations! 

You've journeyed from starting with pytorch to constructing a complete transformer, equipping you the capability to master most of the modern LLMs. 

So in this assignment, we dive into Llama3.2, a SOTA lightweight dense LLM, exploring its end-to-end pipeline, including modeling, inference, and training.

🔥 What'more, as the final assignment, for each basic task, we provide some additional hight-relative tasks but less guided and more challenging, giving you enough room to improvise and explore.


## Debug

*To help you debug:*

### Naive Debug with hard-coded toy cases

* We will give a few test cases with **explicit answers** as toy examples in the visible `test_toy.py` file.
* You had better ensure your code works correctly on your own machine before submitting, with the following command.
    ```sh
    pytest test_toy.py
    ```
* For the final assignment5 sepcifically, run the `test_toy_task{i}.py` or the `test_toy_task{i}.ipynb` for the `i`-th task with the global variable `TEST_WITH_REF` toggled off:
    ```sh
    # TEST_WITH_REF=False
    python test_toy_task{i}.py
    ```

### Deep Debug with reference implementation

* Based on `test_toy.py`, we offer another test file `test_with_ref.py`, in which a **close-sourced reference package** named `ref`, with the same structure as `src`, will be imported (e.g. `from ref import ...`, `from ref.modeling import ...`). Thus you can create your own test cases beyond the toy ones, and compare your answer with it.
* To get access to this `ref`, we provide docker image: [a_env_light:v0](https://hub.docker.com/r/strivin0311/a_env_light), and you can pull it and follow the commands below to run the `test_with_ref.py`.
    ```sh
    # step0. pull the docker image
    docker pull strivin0311/a_env_light:v0

    # step1. run the given script to load the docker image and execute the container
    bash run_docker.sh # or maybe you need run it with sudo

    # step2. get into the repo path mounted into the container
    cd a_repo

    # step3. run the test_with_ref.py
    pytest test_with_ref.py    
    ```
* For the final assignment5 sepcifically, get into the docker container as above and run the `test_toy_task{i}.py` or the `test_toy_task{i}.ipynb` for the `i`-th task with the global variable `TEST_WITH_REF` toggled on:
    ```sh
    # TEST_WITH_REF=True
    python test_toy_task{i}.py
    ```