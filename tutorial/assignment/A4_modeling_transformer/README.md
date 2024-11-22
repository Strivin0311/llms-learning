# assignment-4-modeling-transformer
Finally! 

For this assignment, we are going to bring all the modules we've built in the previous assignments together, and construct a complete decoder-only transformer


## Tasks (100 points)

### Task 1: Transformer Decoder KVCache (20 points)

Please read the description [here](./tasks/task1.md).

### Task 2: Transformer Decoder Layer (50 points)

Please read the description [here](./tasks/task2.md).

### Task 3: Transformer Decoder Block (30 points)

Please read the description [here](./tasks/task3.md).


## Environment

* You should have python 3.10+ installed on your machine.
* (*Optional*) You had better have Nvidia GPU(s) with CUDA12.0+ installed on your machine, otherwise some features may not work properly (*We will do our best to ensure that the difference in hardware does not affect your score.*).
* You are supposed to install all the necessary dependencies with the following command, **which may vary a little among different assignments**.
    ```python
    pip install -r requirements.txt
    ```
* (*Optional*) You are strongly recommended to use a docker image from [Nvidia Pytorch Release](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html) like [23.10](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/rel-23-10.html#rel-23-10) or some newer version as your basic environment in case of denpendency conflicts.


## Submission

* You need to submit your assignment by `git commit` and `git push` this private repository **on the `main` branch** with the specified source files required to fullfill above **before the hard deadline**, otherwise **the delayed assignment will be rejected automatically**.
* Try **NOT to push unnecessary files**, especially some large ones like images, to your repository.
* If you encounter some special problems causing you miss the deadline, please contact the teacher directly (*See [Contact](#contact)*).


## Scoring

* Each assignment will be scored of the points in the range of `0~100` by downloading your code and running the `test_script.sh` script to execute a `test_score.py` file (*invisible to you as empty files*) for some test cases, where the specific files described in [Tasks](#tasks) will be imported locally on our own machine.
* **ALL** the files required to fulfill in [Tasks](#tasks) are under the `src/` directory, which is the **ONLY** directory that will be imported as a **python module**. Therefore, there are several things you should pay attention to:
    * 1. The `__init__.py` is essential for a python module, and we have already initialized all the necessary ones in `src/` for you, so **be careful** when you intend to modify any of them for personal purposes.
    * 2. If you have any other files supposed to be internally imported like `utils.py`, please make sure that they are all under the `src/` directory and **imported relatively** e.g. `from .utils import *`,  `from .common.utils import ...`, etc.
* You will get the maximum score of `100` points if you pass all the tests within the optional time limit.
* You will get the minimum score of `0` points if you fail all the tests within the optional time limit, or run into some exceptions.
* You will get any other score of the points in the middle of `0~100` if you pass part of the tests within the optional time limit, which is the sum of the scores of all the passed test cases as shown in the following table.
    | Test Case | Score | Other Info |
    | --------- | ----- | ---------- |
    | Task1 - Case1 | 10 |  |
    | Task1 - Case2 | 10 |  |
    | Task2 - Case1 | 10 |  |
    | Task2 - Case2 | 10 |  |
    | Task2 - Case3 | 10 |  |
    | Task2 - Case4 | 10 |  |
    | Task2 - Case5 | 10 |  |
    | Task3 - Case1 | 10 |  |
    | Task3 - Case2 | 10 |  |
    | Task3 - Case3 | 10 |  |
    | Total         | 100|  |


## Debug

*To help you debug:*

* 1. Naive debug mode with `test_toy.py`:
    * ① We will give a few test cases with **explicit answers** as toy examples in the visible `test_toy.py` file.
    * ② You had better ensure your code works correctly on your own machine before submitting, with the following command.
        ```sh
        pytest test_toy.py
        ```
    * ③ Feel free to modify the `test_toy.py` file to your specific debugging needs, since we wouldn't run it to score your code, as well as `test_with_ref.py` below.
* 2. Deep debug mode with `test_with_ref.py` given specific docker image (**Unix-like Only**):
    * ① Based on `test_toy.py`, we offer another test file `test_with_ref.py`, in which a **close-sourced reference package** named `ref`, with the same structure as `src`, will be imported (e.g. `from ref import ...`, `from ref.modeling import ...`). Thus you can create your own test cases beyond the toy ones, and compare your answer with it.
    * ② To get access to this `ref`, we provide docker image tar files named like either `a{x}_env_v{y}.tar` (*from ngc pytorch base image*) or `a{x}_env_light_v{y}.tar` (*from ubuntu base image*) uploaded to `NJU Box` (*e.g. the initial version of the light image for the assignment1 will be named `a1_env_light_v0.tar`*), and you can download either one into your private repo.
    * ③ Then follow the example commands below step by step.
        ```sh
        # step0. assumming that the tar file "a{x}_env_light_v{y}.tar" is already downloaded into your private repo
        
        # step1. run the given script to load the docker image (default the light one) and execute the container
        bash run_docker.sh # or maybe you need run it with sudo

        # step2. get into the repo path mounted into the container
        cd a{x}_repo

        # step3. run the test_with_ref.py
        pytest test_with_ref.py
        ```
    * ④ For windows, either you can extract the core docker commands in `run_docker.sh` and run them on your own if you have docker installed, or you can apply some tricks such as [WSL](https://learn.microsoft.com/en-us/windows/wsl/about) or [DinD](https://jpetazzo.github.io/2015/09/03/do-not-use-docker-in-docker-for-ci/) to simulate a unix-like environment.
* 3. Pre-test mode:
    * ① We will pre-test your intermediate submission on the private test cases to score your final submission **as frequently as possible**.
    * ② After each pre-test, we will offer only the score feedback (*See [Feedback](#feedback)*) to allow you to improve your code for higher scores before the hard ddl.
* **Note:** The testing methods in `test_toy.py` and `test_with_ref.py` are provided for debugging purposes only and may not reflect the actual tests used in `test_score.py` for evaluating and scoring your code. So be particularly mindful of handling different situations and maybe some edge cases.


## Feedback

* After scoring your assignment, We will give you a score table like the example one shown below in a new file named `score.md`, by pushing it within a new commit to your repository on a temporary branch called `score-feedback` (*This branch is only for you to view your status on each test cases after every scoring, please do NOT use it for any other purposes*).
    | Test Case | Score | Status | Error Message |
    | --------- | ----- | ------ | ------------- |
    | Task1 - Case1 | 10 | ✅ | |
    | Task1 - Case2 | 10 | ✅ | |
    | Task2 - Case1 | 10 | ✅ | |
    | Task2 - Case2 | 10 | ✅ | |
    | Task2 - Case3 | 10 | ✅ | |
    | Task2 - Case4 | 10 | ✅ | |
    | Task2 - Case5 | 10 | ✅ | |
    | Task3 - Case1 | 10 | ✅ | |
    | Task3 - Case2 | 10 | ✅ | |
    | Task3 - Case3 | 10 | ✅ | |
    | Total         | 100| 😊 | |

* The meaning of the status icons are listed below:
    * ✅: passed the case
    * ❌: failed the case due to wrong answers
    * 🕛: failed the case due to timeout if the time limit is set
    * ❓: failed the case due to some exceptions (the error message will be shown at the corresponding `Error Message` cell)
    * 😊: all passed
    * 🥺: failed at least one case



## Contact

* If you have any questions about the assignment, you can contact the teacher or any assistants directly through QQ group with the number `208283743`.
* You can subscribe to the teacher's bilibili account with UID `390606417` and watch the online courses [here](https://space.bilibili.com/390606417/channel/collectiondetail?sid=3771310).