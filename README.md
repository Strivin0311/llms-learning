# llms-learning 📚 🦙

A repository sharing the literatures and resources about Large Language Models (LLMs) and beyond.

Hope you find this repository useful! 😊


## News 🔥

* [2024.10.24] 
  * Welcome to watch our new online free **LLMs intro course** on [bilibili](https://space.bilibili.com/390606417/channel/collectiondetail?sid=3771310)! 
  * We also open-source and keep updating the [course assignments](./tutorial/assignment/README.md) for you to take a deep dive into LLMs.
  * If you like this course or this repository, you can subscribe to the teacher's [bilibili account](https://space.bilibili.com/390606417) and maybe `star` this GitHub repo 😜.


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
      * [Automatic Mixed-Precision (AMP)](./dev/training/pretrain/amp.md)
      * [Optimizer](./dev/training/pretrain/optimizer.md)
      * [Objectives](./dev/training/pretrain/objective.md)
    * [Fine-Tuning](./dev/training/finetune/)
      * [Efficient Fine-Tuning (EFT)](./dev/training/finetune/peft.md)
      * [Instruction Fine-Tuning (IFT)](./dev/training/finetune/ift.md)
    * [Alignment](./dev/training/alignment/)
      * [Alignment](./dev/training/alignment/align.md)
* [Applications](./app/)
  * [LLMs as Agent](./app/agent/)
  * [LLMs for Autonomous Driving](./app/ads/)
  * [Code LLMs](./app/code/)
  * [Math LLMs](./app/math/)
* [Study](./study/)
  * [Emergence Abilities](./study/emergence.md)
  * [Robustness](./study/robust.md)
  * [Transferability](./study/transfer.md)
  * [In-Context Learning (ICL)](./study/in-context.md)
  * [Long-Context Capabilities](./study/long-context.md)
  * [Empirical Study](./study/empirical.md)
* [Assets](./asset/)


---

**Note**:

* Each markdown file contains collected papers roughly sorted by `published year` in descending order; in other words, newer papers are generally placed at the top. However, this arrangement is not guaranteed to be completely accurate, as the `published year` may not always be clear.

* The taxonomy is complex and not strictly orthogonal, so don't be surprised if the same paper appears multiple times under different tracks.
