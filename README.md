# llms-learning

A repository sharing the literatures and resources about Large Language Models (LLMs)


## Table of Contents

* [Development Stages](./dev/)
  * [Modeling](./dev/modeling/)
    * [Mixture of Experts (MoE)](./dev/modeling/moe.md)
    * [Weight Quantization](./dev/compress/quantize.md)
    * [Model Pruning](./dev/compress/prune.md)
    * [Model Editing](./dev/modeling/edit.md)
    * [Activation Functions](./dev/modeling/afunc.md)
    * [LLMs Alternatives](./dev/modeling/alt.md)
      * [SSMs](./dev/modeling/alt.md#ssms)
      * [LongConv](./dev/modeling/alt.md#longconv)
      * [Miscellaneous](./dev/modeling/alt.md#miscellaneous)
  * [Training](./dev/train/)
    * [Pre-Training](./dev/train/pretrain.md)
      * [Efficient Pre-Training](./dev/train/pretrain.md#efficient-pretraining)
      * [Effective Pre-Training](./dev/train/pretrain.md#effective-pretraining)
      * [Pretraining Corpus](./dev/train/pretrain.md#pretraining-corpus)
      * [Pretraining Objectives](./dev/train/pretrain.md#pretraining-objectives)
    * [Parallelism](./dev/train/parallel.md)
      * [Integration of Parallelism](./dev/train/parallel.md#integration-of-parallelism)
      * [Expert Parallelism (EP)](./dev/train/parallel.md#expert-parallelism-ep)
      * [Context Parallelism (CP)](./dev/train/parallel.md#context-parallelism-cp)
      * [Pipeline Parallelism (PP)](./dev/train/parallel.md#pipeline-parallelism-pp)
      * [Sequence Parallelism (SP)](./dev/train/parallel.md#sequence-parallelism-sp)
      * [Tensor Parallelism (TP)](./dev/train/parallel.md#tensor-parallelism-tp)
      * [Data Parallelism (DP)](./dev/train/parallel.md#data-parallelism-dp)
    * [Automatic Mixed-Precision](./dev/train/amp.md)
    * [Optimizer](./dev/train/optimizer.md)
    * [Offloading](./dev/train/offload.md)
  * [Fine-Tuning](./dev/finetune/)
    * [Efficient Fine-Tuning](./dev/finetune/peft.md)
      * [PEFT](./dev/finetune/peft.md#parameter-efficient-fine-tuning-peft)
    * [Instruction Fine-Tuning (IFT)](./dev/finetune/instruction.md)
    * [Alignment](./dev/finetune/alignment.md)
  * [Post-Training](./dev/post-train/)
    * [Inference](./dev/post-train/inference.md)
      * [Efficient Inference](./dev/post-train/inference.md#efficient-inference)
      * [Effective Decoding](./dev/post-train/inference.md#effective-decoding)
      * [Calibration](./dev/post-train/inference.md#calibration)
    * [Evaluation](./dev/post-train/evaluate.md)
      * [Benchmarking](./dev/post-train/evaluate.md#benchmarking)
      * [English Benchmarks](./dev/post-train/evaluate.md#english-benchmarks)
      * [Chinese Benchmarks](./dev/post-train/evaluate.md#chinese-benchmarks)
      * [Multi-Language Benchmarks](./dev/post-train/evaluate.md#multi-language-benchmarks)
      * [Metrics](./dev/post-train/evaluate.md#metrics)
    * [Deployment](./dev/post-train/deploy.md)
* [Abilities](./abilities/)
  * [In-Context Learning (ICL)](./abilities/in-context.md)
  * [Long-Context Capabilities](./abilities/long-context.md)
  * [Multi-Modal Capabilities](./abilities/multi-modal.md)
  * [Reasoning](./abilities/reasoning.md)
  * [Emergence Abilities](./abilities/emergence.md)
  * [Robustness](./abilities/robust.md)
* [Applications](./app/)
  * [LLMs as Agent](./app/agent.md)
  * [LLMs for Embodied AI](./app/embodied.md)
  * [LLMs powered Generation](./app/gen.md)
  * [LLMs as Tool Caller](./app/tool.md)
* [Model Hubs](./model-hubs)
  * [Foundation Models](./model-hubs/foundation)
    * [Baichuan](./model-hubs/foundation/baichuan.md)
    * [BART](./model-hubs/foundation/bart.md)
    * [BERT](./model-hubs/foundation/bert.md)
    * [Bloom](./model-hubs/foundation/bloom.md)
    * [Claude](./model-hubs/foundation/claude.md)
    * [DeepSeek](./model-hubs/foundation/deepseek.md)
    * [ERNIE](./model-hubs/foundation/ernie.md)
    * [Gemini](./model-hubs/foundation/gemini.md)
    * [GLM](./model-hubs/foundation/glm.md)
    * [GPT](./model-hubs/foundation/gpt.md)
    * [Hunyuan](./model-hubs/foundation/hunyuan.md)
    * [Llama](./model-hubs/foundation/llama.md)
    * [Mistral](./model-hubs/foundation/mistral.md)
    * [PaLM](./model-hubs/foundation/palm.md)
    * [Qwen](./model-hubs/foundation/qwen.md)
    * [Skywork](./model-hubs/foundation/skywork.md)
    * [T5](./model-hubs/foundation/t5.md)
    * [Miscellaneous](./model-hubs/foundation/miscellaneous.md)
  * [Domain Specific Models](./model-hubs/domain-specific)
    * [Code LLMs](./model-hubs/domain-specific/code.md)
    * [Math LLMs](./model-hubs/domain-specific/math.md)
    * [Financial LLMs](./model-hubs/domain-specific/finance.md)
* [Empirical Study (Surveys)](./empirical.md)
* [Tutorials (TODO ...)](./tutorial.md)


## More to Learn

* This repo is a sub-repo of [dgm-learning](https://github.com/Strivin0311/ai-learning/tree/main/techs/dl/dgm) sharing a broader literature about Deep Generative Models (DGMs) including Diffusers, SSMs, GANs, VAEs, etc, as a subtrack of my root repo [ai-learning](https://github.com/Strivin0311/ai-learning), which is a general resource hub (mainly papers and codes) for all ML-based AI technologies and applications.

* Moreover, I've also built another sub-repo [long-llms-learning](https://github.com/Strivin0311/long-llms-learning), to specifically share the literature about how to activate the potential of LLMs'  long-context capabilities across all stages from modeling, training to inference and evaluation, with a [survey](https://arxiv.org/abs/2311.12351) attached.

* Last but not least, recently I've also built another sub-repo [mmlms-learning](https://github.com/Strivin0311/mmlms-learning), to specifically share the literature about Multi-Modal Language Modeles (MMLMs), which, as I believe, is the most intriguing field of AGI.



---

**Note**:
* In each markdown file, the collected paper may be roughly sorted by the `published year` in descending order, i.e. the newer the paper, the topper it will be put on the file, but it's not one-hundred percent sure since the `published year` is not always clear.
* The taxonomy is too complicated to be orthogonal, so don't be confused when the same paper is collected in different tracks for many times.
