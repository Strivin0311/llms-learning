# Inference on LLMs
*Here're some resources about Inference on LLMs*


### Efficient Inference


#### 1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs

tag: `BitNet 1.1`

paper link: [here](https://arxiv.org/pdf/2410.16144v2)

github link: [here](https://github.com/microsoft/bitnet)

citation:

```bibtex
@misc{wang20241bitaiinfra11,
      title={1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs}, 
      author={Jinheng Wang and Hansong Zhou and Ting Song and Shaoguang Mao and Shuming Ma and Hongyu Wang and Yan Xia and Furu Wei},
      year={2024},
      eprint={2410.16144},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.16144}, 
}
```


#### Fast Inference of Mixture-of-Experts Language Models with Offloading

paper link: [here](https://arxiv.org/pdf/2312.17238.pdf)

github link: [here](https://github.com/dvmazur/mixtral-offloading)

citation:

```bibtex
@misc{eliseev2023fast,
      title={Fast Inference of Mixture-of-Experts Language Models with Offloading}, 
      author={Artyom Eliseev and Denis Mazur},
      year={2023},
      eprint={2312.17238},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

#### PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU

paper link: [here](https://ipads.se.sjtu.edu.cn/_media/publications/powerinfer-20231219.pdf)

citation:

```bibtex
@misc{song2023powerinfer,
      title={PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU}, 
      author={Yixin Song and Zeyu Mi and Haotong Xie and Haibo Chen},
      year={2023},
      eprint={2312.12456},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

#### LLM in a flash: Efficient Large Language Model Inference with Limited Memory

paper link: [here](https://arxiv.org/pdf/2312.11514.pdf)

citation:

```bibtex
@article{alizadeh2023llm,
  title={LLM in a flash: Efficient Large Language Model Inference with Limited Memory},
  author={Alizadeh, Keivan and Mirzadeh, Iman and Belenko, Dmitry and Khatamifard, Karen and Cho, Minsik and Del Mundo, Carlo C and Rastegari, Mohammad and Farajtabar, Mehrdad},
  journal={arXiv preprint arXiv:2312.11514},
  year={2023}
}
```

#### S-LoRA: Serving Thousands of Concurrent LoRA Adapters

paper link: [here](https://arxiv.org/pdf/2311.03285)

citation:

```bibtex
@article{sheng2023s,
  title={S-LoRA: Serving Thousands of Concurrent LoRA Adapters},
  author={Sheng, Ying and Cao, Shiyi and Li, Dacheng and Hooper, Coleman and Lee, Nicholas and Yang, Shuo and Chou, Christopher and Zhu, Banghua and Zheng, Lianmin and Keutzer, Kurt and others},
  journal={arXiv preprint arXiv:2311.03285},
  year={2023}
}
```

#### Punica: Multi-Tenant LoRA Serving

paper link: [here](https://arxiv.org/pdf/2310.18547.pdf)

citation:

```bibtex
@misc{chen2023punica,
      title={Punica: Multi-Tenant LoRA Serving}, 
      author={Lequn Chen and Zihao Ye and Yongji Wu and Danyang Zhuo and Luis Ceze and Arvind Krishnamurthy},
      year={2023},
      eprint={2310.18547},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```

#### CacheGen: Fast Context Loading for Language Model Applications

paper link: [here](https://arxiv.org/pdf/2310.07240)

citation:

```bibtex
@article{liu2023cachegen,
  title={CacheGen: Fast Context Loading for Language Model Applications},
  author={Liu, Yuhan and Li, Hanchen and Du, Kuntai and Yao, Jiayi and Cheng, Yihua and Huang, Yuyang and Lu, Shan and Maire, Michael and Hoffmann, Henry and Holtzman, Ari and others},
  journal={arXiv preprint arXiv:2310.07240},
  year={2023}
}
```


#### SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills

paper link: [here](https://arxiv.org/pdf/2308.16369)

citation:

```bibtex
@misc{agrawal2023sarathiefficientllminference,
      title={SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills}, 
      author={Amey Agrawal and Ashish Panwar and Jayashree Mohan and Nipun Kwatra and Bhargav S. Gulavani and Ramachandran Ramjee},
      year={2023},
      eprint={2308.16369},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2308.16369}, 
}
```


#### DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference

blog link: [here](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen)

github link: [here](https://github.com/microsoft/DeepSpeed-MII)

tutorial link: [here](../../tutorial/notebook/tutorial_deepspeed_infer.ipynb)

citation:

```bibtex
@misc{DeepSpeed2023FastGen,
  author = {DeepSpeed Team},
  title = {DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference},
  year = {2023},
  month = {Nov},
  howpublished = {\url{https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen}},
}
```


#### AirLLM: inference 70B LLM with 4GB single GPU

github link: [here](https://github.com/lyogavin/Anima/tree/main/air_llm)

tutorial link: [here](../../tutorial/notebook/tutorial_airllm.ipynb)

citation
```bibtex
@software{airllm2023,
  author = {Gavin Li},
  title = {AirLLM: scaling large language models on low-end commodity computers},
  url = {https://github.com/lyogavin/Anima/tree/main/air_llm},
  version = {0.0},
  year = {2023},
}
```


#### High-throughput generative inference of large language models with a single gpu

paper link: [here](https://arxiv.org/pdf/2303.06865)

citation:

```bibtex
@article{sheng2023high,
  title={High-throughput generative inference of large language models with a single gpu},
  author={Sheng, Ying and Zheng, Lianmin and Yuan, Binhang and Li, Zhuohan and Ryabinin, Max and Fu, Daniel Y and Xie, Zhiqiang and Chen, Beidi and Barrett, Clark and Gonzalez, Joseph E and others},
  journal={arXiv preprint arXiv:2303.06865},
  year={2023}
}
```

#### ZeRO-Inference: Democratizing massive model inference

blog link: [here](https://www.deepspeed.ai/2022/09/09/zero-inference.html)

github link: [here](https://github.com/microsoft/DeepSpeed/)

citation:

```bibtex
@misc{Zero2022Inference,
  author = {DeepSpeed Team},
  title = {ZeRO-Inference: Democratizing massive model inference},
  year = {2022},
  month = {Sep},
  howpublished = {\url{https://www.deepspeed.ai/2022/09/09/zero-inference.html}},
}
```

#### Orca: A Distributed Serving System for Transformer-Based Generative Models

paper link: [here](https://www.usenix.org/system/files/osdi22-yu.pdf)

citation:

```bibtex
@inproceedings {280922,
  author = {Gyeong-In Yu and Joo Seong Jeong and Geon-Woo Kim and Soojeong Kim and Byung-Gon Chun},
  title = {Orca: A Distributed Serving System for {Transformer-Based} Generative Models},
  booktitle = {16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22)},
  year = {2022},
  isbn = {978-1-939133-28-1},
  address = {Carlsbad, CA},
  pages = {521--538},
  url = {https://www.usenix.org/conference/osdi22/presentation/yu},
  publisher = {USENIX Association},
  month = jul
}
```


#### A BetterTransformer for Fast Transformer Inference

blog link: [here](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/)

homepage link: [here](https://huggingface.co/docs/optimum/bettertransformer/overview)

tutorial link: [here](../../tutorial/notebook/BetterTransformerDemo.ipynb)

supported model list link: [here](https://huggingface.co/docs/optimum/bettertransformer/overview#supported-models)

citation:

```bibtex
@online{bettertransformer,
  author = {Michael Gschwind, Eric Han, Scott Wolchok, Rui Zhu, Christian Puhrsch},
  title = {A Better Transformer for Fast Transformer Inference},
  year = {2022},
  month = {July},
  url = {\url{https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/}}
}
```

#### DeepSpeed Inference: Multi-GPU inference with customized inference kernels and quantization support

blog link: [here](https://www.deepspeed.ai/2021/03/15/inference-kernel-optimization.html)

github link: [here](https://github.com/microsoft/DeepSpeed/)

citation:

```bibtex
@misc{DeepSpeed2021InferenceKernelOptimization,
  author = {DeepSpeed Team},
  title = {DeepSpeed Inference: Multi-GPU inference with customized inference kernels and quantization support},
  year = {2021},
  month = {March},
  howpublished = {\url{https://www.deepspeed.ai/2021/03/15/inference-kernel-optimization.html}},
}
```


#### Fast transformer decoding: One write-head is all you need

paper link: [here](https://arxiv.org/pdf/1911.02150.pdf)

citation:

```bibtex
@article{shazeer2019fast,
  title={Fast transformer decoding: One write-head is all you need},
  author={Shazeer, Noam},
  journal={arXiv preprint arXiv:1911.02150},
  year={2019}
}
```

### Effective Decoding


#### The Consensus Game: Language Model Generation via Equilibrium Search

paper link: [here](https://arxiv.org/pdf/2310.09139.pdf)

citation:

```bibtex
@misc{jacob2023consensus,
      title={The Consensus Game: Language Model Generation via Equilibrium Search}, 
      author={Athul Paul Jacob and Yikang Shen and Gabriele Farina and Jacob Andreas},
      year={2023},
      eprint={2310.09139},
      archivePrefix={arXiv},
      primaryClass={cs.GT}
}
```

#### Fast Inference from Transformers via Speculative Decoding

tag: `Speculative Decoding`

paper link: [here](https://arxiv.org/pdf/2211.17192v2)

github link: [here](https://github.com/lucidrains/speculative-decoding)

citation:

```bibtex
@misc{leviathan2023fastinferencetransformersspeculative,
      title={Fast Inference from Transformers via Speculative Decoding}, 
      author={Yaniv Leviathan and Matan Kalman and Yossi Matias},
      year={2023},
      eprint={2211.17192},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2211.17192}, 
}
```

#### Contrastive Decoding: Open-ended Text Generation as Optimization

paper link: [here](https://arxiv.org/pdf/2210.15097.pdf)

citation:

```bibtex
@misc{li2023contrastive,
      title={Contrastive Decoding: Open-ended Text Generation as Optimization}, 
      author={Xiang Lisa Li and Ari Holtzman and Daniel Fried and Percy Liang and Jason Eisner and Tatsunori Hashimoto and Luke Zettlemoyer and Mike Lewis},
      year={2023},
      eprint={2210.15097},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### The Curious Case of Neural Text Degeneration (Nucleus Sampling)

paper link: [here](https://arxiv.org/pdf/2209.11057.pdf)

citation:

```bibtex
@misc{holtzman2020curious,
      title={The Curious Case of Neural Text Degeneration}, 
      author={Ari Holtzman and Jan Buys and Li Du and Maxwell Forbes and Yejin Choi},
      year={2020},
      eprint={1904.09751},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
    
    


### Calibration


#### Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering (BC)

paper link: [here](https://arxiv.org/pdf/2309.17249.pdf)

citation:

```bibtex
@misc{zhou2024batch,
      title={Batch Calibration: Rethinking Calibration for In-Context Learning and Prompt Engineering}, 
      author={Han Zhou and Xingchen Wan and Lev Proleev and Diana Mincu and Jilin Chen and Katherine Heller and Subhrajit Roy},
      year={2024},
      eprint={2309.17249},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


####  Mitigating label biases for in-context learning (DC)

paper link: [here](https://arxiv.org/pdf/2305.19148.pdf)

citation:

```bibtex
@misc{fei2023mitigating,
      title={Mitigating Label Biases for In-context Learning}, 
      author={Yu Fei and Yifan Hou and Zeming Chen and Antoine Bosselut},
      year={2023},
      eprint={2305.19148},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### Prototypical Calibration for Few-shot Learning of Language Models (PC)

paper link: [here](https://arxiv.org/pdf/2205.10183.pdf)

citation:

```bibtex
@misc{han2022prototypical,
      title={Prototypical Calibration for Few-shot Learning of Language Models}, 
      author={Zhixiong Han and Yaru Hao and Li Dong and Yutao Sun and Furu Wei},
      year={2022},
      eprint={2205.10183},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### Calibrate before use: Improving few-shot performance of language models (CC)

paper link: [here](http://proceedings.mlr.press/v139/zhao21c/zhao21c.pdf)

citation:

```bibtex
@inproceedings{zhao2021calibrate,
  title={Calibrate before use: Improving few-shot performance of language models},
  author={Zhao, Zihao and Wallace, Eric and Feng, Shi and Klein, Dan and Singh, Sameer},
  booktitle={International Conference on Machine Learning},
  pages={12697--12706},
  year={2021},
  organization={PMLR}
}
```

