# llms-learning

A repository sharing the literatures about Transformer-based large language models (LLMs)



## More to Learn

* This repo is a sub-track for my [dgm-learning](https://github.com/Strivin0311/dgm-learning) repo, where you can learn more technologies about the deep generative models
* I've also built another repo [long-llms-learning](https://github.com/Strivin0311/long-llms-learning), to specifically share the literature about how to model and evaluate the long-context capabilities of LLMs, with a survey paper attached


## Table of Contents

* [Model Hubs](./models)
  * [BERT](./models/bert.md)
  * [ERNIE](./models/ernie.md)
  * [GLM](./models/glm.md)
  * [GPT](./models/gpt.md)
  * [Llama](./models/llama.md)
  * [Miscellaneous](./models/miscellaneous.md)
* [Modeling Stages](./modeling/)
  * [Training](./modeling/train/)
    * [Pre-Training](./modeling/train/pretrain.md)
      * [Efficient Pre-Training](./modeling/train/pretrain.md#efficient-pretraining)
      * [Effective Pre-Training](./modeling/train/pretrain.md#effective-pretraining)
      * [Pretraining Corpus](./modeling/train/pretrain.md#pretraining-corpus)
    * [Parallelism](./modeling/train/parallel.md)
  * [Fine-Tuning](./modeling/finetune/)
    * [Efficient Fine-Tuning](./modeling/finetune/peft.md)
      * [PEFT](./modeling/finetune/peft.md#parameter-efficient-fine-tuning-peft)
    * [Instruction Fine-Tuning (IFT)](./modeling/finetune/instruction.md)
    * [Alignment](./modeling/finetune/alignment.md)
  * [Post-Training](./modeling/post-train/)
    * [Inference](./modeling/post-train/inference.md)
      * [Efficient Inference](./modeling/post-train/inference.md#efficient-inference)
      * [Effective Inference](./modeling/post-train/inference.md#effective-inference)
      * [Calibration](./modeling/post-train/inference.md#calibration)
    * [Evaluation](./modeling/post-train/eval.md)
    * [Deployment](./modeling/post-train/deploy.md)
    * [Decoding](./modeling/post-train/decode.md)
  * [Weight Compression](./modeling/weight-compress/)
    * [Pruning](./modeling/weight-compress/pruning.md)
    * [Quantization](./modeling/weight-compress/quantization.md)
  * [Architecture](./modeling/architecture/)
    * [Mixture of Experts (MoE)](./modeling/architecture/mixture-of-experts.md)
    * [Activation Functions](./modeling/architecture/activation-func.md)
* [Abilities](./abilities/)
  * [In-Context Learning (ICL)](./abilities/in-context.md)
  * [Long context](./abilities/long-context.md)
  * [Multi Modal](./abilities/multi-modal.md)
  * [Reasoning](./abilities/reasoning.md)
  * [Emergence](./abilities/emergence.md)
  * [Robustness](./abilities/robust.md)
  * [Model Editing](./abilities/edit.md)
* [Applications](./app/)
  * [LLMs as Agent](./app/agent.md)
  * [LLMs for Embodied AI](./app/embodied.md)
  * [LLMs powered Generation](./app/gen.md)
  * [LLMs as Tool Caller](./app/tool.md)
* [Survey](./survey.md)
* [Tutorials](./tutorial.md)


