# llms-learning 📚 🦙

A repository sharing the literatures and resources about Large Language Models (LLMs) and beyond.

Hope you find this repository useful! 😊


## News 🔥

* **2024.10.24**
  * Welcome to watch our new online free **LLMs intro course** on [bilibili](https://space.bilibili.com/390606417/channel/collectiondetail?sid=3771310)! 
  * We also open-source and keep updating the [course assignments](./tutorial/assignment/README.md) for you to take a deep dive into LLMs.
  * If you like this course or this repository, you can subscribe to the teacher's [bilibili account](https://space.bilibili.com/390606417) and maybe ⭐ this GitHub repo 😜.
* **2024.03.07**
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
        * [Dense](./dev/modeling/lm/transformer/dense.md)
        * [Sparse, Mixture of Experts (MoE)](./dev/modeling/lm/transformer/sparse.md)
        * [Normalization](./dev/modeling/lm/transformer/normalization.md)
        * [Position Embedding](./dev/modeling/lm/transformer/pe.md)
        * [Activation Functions](./dev/modeling/lm/transformer/act_func.md)
      * [State Space Models (SSM)](./dev/modeling/lm/ssm.md)
      * [Kolmogorov-Arnold Networks (KAN)](./dev/modeling/lm/kan.md)
      * [Miscellaneous](./dev/modeling/lm/misc.md)
    * [Multi-Modal Language Models](./dev/modeling/mm/)
      * [Text <-> Image](./dev/modeling/mm/t2i.md)
      * [Text <-> Audio](./dev/modeling/mm/t2a.md)
      * [Text <-> Video](./dev/modeling/mm/t2v.md)
      * [Text <-> Omni](./dev/modeling/mm/t2o.md)
  * [Inference](./dev/inference/)
    * [Serving](./dev/serving/serving.md)
    * [Quantization](./dev/serving/quantize.md)
    * [Pruning](./dev/serving/prune.md)
    * [Decoding](./dev/serving/decode.md)
    * [Evaluation](./dev/serving/evaluate.md)
  * [Training](./dev/training/)
    * [Pre-Training](./dev/training/pretrain)
      * [General Training Recipes](./dev/training/pretrain/recipe.md)
      * [Mixed-Precision Training](./dev/training/pretrain/mpt.md)
      * [Heterogenous Training](./dev/training/pretrain/hetero.md)
      * [Parallelism](./dev/training/pretrain/parallel.md)
        * [Integration of Parallelism (IP)](./dev/training/pretrain/parallelism/ip.md)
        * [Data Parallelism (DP)](./dev/training/pretrain/parallelism/dp.md)
        * [Tensor Parallelism (TP)](./dev/training/pretrain/parallelism/tp.md)
        * [Pipeline Parallelism (PP)](./dev/training/pretrain/parallelism/pp.md)
        * [Context Parallelism (CP)](./dev/training/pretrain/parallelism/cp.md)
        * [Expert Parallelism (EP)](./dev/training/pretrain/parallelism/ep.md)
        * [Automatic Parallelism (AP)](./dev/training/pretrain/parallelism/ap.md)
        * [Distributed Communication](./dev/training/pretrain/parallelism/comm.md)
      * [Memory Management](./dev/training/pretrain/mem_manage/)
        * [Recomputation / Activation Checkpointing](./dev/training/pretrain/mem_manage/recomp.md)
        * [Offloading](./dev/training/pretrain/mem_manage/offload.md)
        * [Checkpointing](./dev/training/pretrain/mem_manage/ckpt.md)
        * [Device Placement](./dev/training/pretrain/mem_manage/dev_place.md)
      * [Optimizer](./dev/training/pretrain/optimizer.md)
      * [Objectives / Loss Functions](./dev/training/pretrain/objective.md)
    * [Fine-Tuning](./dev/training/finetune/)
      * [Supervised Fine-Tuning (SFT)](./dev/training/finetune/sft.md)
      * [Alignment Fine-Tuning (AFT)](./dev/training/finetune/align.md)
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
