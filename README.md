# llms-learning 📚 🦙

A repository sharing the literatures and resources about Large Language Models (LLMs) and beyond.

Hope you find this repository useful! 😊


## News 🔥

* [2024.10.24] 
  * Welcome to watch our new online free **LLMs intro course** on [bilibili](https://space.bilibili.com/390606417/channel/collectiondetail?sid=3771310)! 
  * We also open-source and keep updating the [course assignments](./tutorial/assignment/README.md) for you to take a deep dive into LLMs.
  * If you like this course or this repository, you can subscribe to the teacher's [bilibili account](https://space.bilibili.com/390606417) and maybe `star` this GitHub repo 😜.
* [2024.03.07]
  * We offer a comprehensive [notebook tutorial](./tutorial/notebook/tutorial_triton.ipynb) on efficient GPU kernel coding using [Triton](https://github.com/triton-lang/triton), building upon the [official tutorials](https://triton-lang.org/main/getting-started/tutorials/index.html) and extending them with additional hands-on examples, such as the **Flash Attention 2 forward/backward kernel**.
  * In addition, we also provide a step-by-step [math derivation](./dev/modeling/lm/transformer/attn/fa2_deri.md) of [Flash Attention 2](https://arxiv.org/abs/2307.08691), enabling a deeper understanding of its underlying mechanics.


## Table of Contents

* [Tutorials](./tutorial/)
  * [Notebooks](./tutorial/notebook/)
  * [LLMs Intro Course Assignments](./tutorial/assignment/README.md)
* [Development Stages](./dev/)
  * [Modeling](./dev/modeling/)
    * [Language Models](./dev/modeling/lm)
      * [Transformer](./dev/modeling/lm/transformer/)
        * [Attention](./dev/modeling/lm/transformer/attn/)
          * [Full Attention](./dev/modeling/lm/transformer/attn/full_attn.md)
          * [Sparse Attention](./dev/modeling/lm/transformer/attn/sparse_attn.md)
          * [Linear Attention](./dev/modeling/lm/transformer/attn/linear_attn.md)
        * [Mixture of Experts (MoE)](./dev/modeling/lm/transformer/moe.md)
        * [Normalization](./dev/modeling/lm/transformer/normalization.md)
        * [Position Embedding](./dev/modeling/lm/transformer/pe.md)
        * [Activation Functions](./dev/modeling/lm/transformer/act_func.md)
      * [State Space Models (SSM)](./dev/modeling/lm/ssm.md)
      * [Miscellaneous](./dev/modeling/lm/misc.md)
    * [Multi-Modal Language Models](./dev/modeling/mm/)
      * [Text <-> Image](./dev/modeling/mm/t2i.md)
      * [Text <-> Audio](./dev/modeling/mm/t2a.md)
      * [Text <-> Video](./dev/modeling/mm/t2v.md)
      * [Text <-> Omni](./dev/modeling/mm/t2o.md)
  * [Serving](./dev/serving/)
    * [Inference](./dev/serving/inference.md)
    * [Quantization](./dev/serving/quantize.md)
    * [Pruning](./dev/serving/prune.md)
    * [Deployment](./dev/serving/deploy.md)
    * [Evaluation](./dev/serving/evaluate.md)
  * [Training](./dev/training/)
    * [Pre-Training](./dev/training/pretrain)
      * [Pre-Training](./dev/training/pretrain/pretrain.md)
      * [Parallelism](./dev/training/pretrain/parallel.md)
        * [Integration of Parallelism (IP)](./dev/training/pretrain/parallel.md#integration-of-parallelism)
        * [Expert Parallelism (EP)](./dev/training/pretrain/parallel.md#expert-parallelism-ep)
        * [Context Parallelism (CP)](./dev/training/pretrain/parallel.md#context-parallelism-cp)
        * [Pipeline Parallelism (PP)](./dev/training/pretrain/parallel.md#pipeline-parallelism-pp)
        * [Sequence Parallelism (SP)](./dev/training/pretrain/parallel.md#sequence-parallelism-sp)
        * [Tensor Parallelism (TP)](./dev/training/pretrain/parallel.md#tensor-parallelism-tp)
        * [Data Parallelism (DP)](./dev/training/pretrain/parallel.md#data-parallelism-dp)
      * [Offloading](./dev/training/pretrain/offload.md)
      * [Checkpointing](./dev/training/pretrain/ckpt.md)
      * [Mixed-Precision](./dev/training/pretrain/mp.md)
      * [Optimizer](./dev/training/pretrain/optimizer.md)
      * [Objectives / Loss Functions](./dev/training/pretrain/objective.md)
    * [Fine-Tuning](./dev/training/finetune/)
      * [Efficient Fine-Tuning (EFT)](./dev/training/finetune/peft.md)
      * [Instruction Fine-Tuning (IFT)](./dev/training/finetune/ift.md)
    * [Alignment](./dev/training/alignment/)
      * [Alignment](./dev/training/alignment/align.md)
* [Applications](./app/)
  * [Retrieval-Augmented Generation (RAG)](./app/rag.md)
  * [LLMs as Agents](./app/agent.md)
  * [LLMs for Autonomous Driving](./app/auto_drive.md)
  * [LLMs for Code](./app/code.md)
  * [LLMs for Math](./app/math.md)
* [Study](./study/)
  * [Emergence Abilities](./study/emergence.md)
  * [Embodied Intelligence](./study/embodied.md)
  * [Robustness](./study/robust.md)
  * [Transferability](./study/transfer.md)
  * [In-Context Learning (ICL)](./study/in-context.md)
  * [Long-Context Capabilities](./study/long-context.md)
  * [General Empirical Study](./study/empirical.md)
  * [General Surveys](./study/survey.md)
* [Basics](./base/)
* [Assets](./asset/)


---

**Note**:

* Each markdown file contains collected papers roughly sorted by `published year` in descending order; in other words, newer papers are generally placed at the top. However, this arrangement is not guaranteed to be completely accurate, as the `published year` may not always be clear.

* The taxonomy is complex and not strictly orthogonal, so don't be surprised if the same paper appears multiple times under different tracks.
