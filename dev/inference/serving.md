# Serving on LLMs
*Here're some resources about Serving on LLMs, i.e. Service-Level Inference on LLMs*


## Method


#### FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving

tag: `FlashInfer` | `MLSys25` | `Nvidia` | `University of Washington`

paper link: [here](https://arxiv.org/pdf/2501.01005)

code link: [here](https://github.com/flashinfer-ai/flashinfer/)

homepage link: [here](https://flashinfer.ai/)

citation:

```bibtex
@misc{ye2025flashinferefficientcustomizableattention,
      title={FlashInfer: Efficient and Customizable Attention Engine for LLM Inference Serving}, 
      author={Zihao Ye and Lequn Chen and Ruihang Lai and Wuwei Lin and Yineng Zhang and Stephanie Wang and Tianqi Chen and Baris Kasikci and Vinod Grover and Arvind Krishnamurthy and Luis Ceze},
      year={2025},
      eprint={2501.01005},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2501.01005}, 
}
```


#### 1-bit AI Infra: Part 1.1, Fast and Lossless BitNet b1.58 Inference on CPUs

tag: `BitNet.cpp` | `BitNet b1.58` | `Microsoft`

paper link: [here](https://arxiv.org/pdf/2410.16144)

code link: [here](https://github.com/microsoft/bitnet)

homepage link: [here](https://thegenerality.com/agi/)

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


#### MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention

tag: `MInference 1.0` | `Dynamic Sparse Attention` | `NIPS24` | `Microsoft`

paper link: [here](https://arxiv.org/pdf/2407.02490)

code link: [here](https://github.com/microsoft/MInference)

homepage link: [here](https://aka.ms/MInference)

citation:

```bibtex
@misc{jiang2024minference10acceleratingprefilling,
      title={MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention}, 
      author={Huiqiang Jiang and Yucheng Li and Chengruidong Zhang and Qianhui Wu and Xufang Luo and Surin Ahn and Zhenhua Han and Amir H. Abdi and Dongsheng Li and Chin-Yew Lin and Yuqing Yang and Lili Qiu},
      year={2024},
      eprint={2407.02490},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.02490}, 
}
```


#### PowerInfer-2: Fast Large Language Model Inference on a Smartphone

tag: `PowerInfer-2` | `IPADS` | `SJTU`

paper link: [here](https://arxiv.org/pdf/2410.11298)

code link: [here](https://github.com/SJTU-IPADS/PowerInfer)

citation:

```bibtex
@misc{xue2024powerinfer2fastlargelanguage,
      title={PowerInfer-2: Fast Large Language Model Inference on a Smartphone}, 
      author={Zhenliang Xue and Yixin Song and Zeyu Mi and Xinrui Zheng and Yubin Xia and Haibo Chen},
      year={2024},
      eprint={2406.06282},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.06282}, 
}
```


#### Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve

tag: `Sarathi-Serve` | `OSDI24` | `Microsoft`

paper link: [here](https://www.usenix.org/system/files/osdi24-agrawal.pdf)

code link: [here](https://github.com/microsoft/sarathi-serve)

citation:

```bibtex
@misc{agrawal2024tamingthroughputlatencytradeoffllm,
      title={Taming Throughput-Latency Tradeoff in LLM Inference with Sarathi-Serve}, 
      author={Amey Agrawal and Nitin Kedia and Ashish Panwar and Jayashree Mohan and Nipun Kwatra and Bhargav S. Gulavani and Alexey Tumanov and Ramachandran Ramjee},
      year={2024},
      eprint={2403.02310},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2403.02310}, 
}
```


#### Fast Inference of Mixture-of-Experts Language Models with Offloading

tag: `Expert Offloading` | `Mixtral` | `MoE`

paper link: [here](https://arxiv.org/pdf/2312.17238.pdf)

code link: [here](https://github.com/dvmazur/mixtral-offloading)

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

tag: `PowerInfer` | `SOSP24` | `IPADS` | `SJTU`

paper link: [here](https://ipads.se.sjtu.edu.cn/_media/publications/powerinfer-20231219.pdf)

code link: [here](https://github.com/SJTU-IPADS/PowerInfer)

followup work: [here](https://arxiv.org/pdf/2410.11298)

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

tag: `LLM in a flash` | `ACL24` | `Apple`

paper link: [here](https://aclanthology.org/2024.acl-long.678.pdf)

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

tag: `S-LoRA` | `MLSys24` | `UC Berkeley` | `Stanford University`

paper link: [here](https://arxiv.org/pdf/2311.03285)

code link: [here](https://github.com/S-LoRA/S-LoRA)

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

tag: `Punica` | `MLSys24` | `University of Washington`

paper link: [here](https://arxiv.org/pdf/2310.18547.pdf)

code link: [here](https://github.com/punica-ai/punica)

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

tag: `CacheGen` | `ACM SIGCOMM24` | `Microsoft` | `University of Chicago`

paper link: [here](https://arxiv.org/pdf/2310.07240)

code link: [here](https://github.com/UChi-JCL/CacheGen)

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

tag: `Sarathi` | `Chunked Prefill` | `Microsoft`

paper link: [here](https://arxiv.org/pdf/2308.16369)

follow-up link: [here](https://www.usenix.org/system/files/osdi24-agrawal.pdf)

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

tag: `DeepSpeed-FastGen` | `MII` | `DeepSpeed` | `Microsoft`

blog link: [here](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen)

code link: [here](https://github.com/microsoft/DeepSpeed-MII)

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


#### FlexGen: High-Throughput Generative Inference of Large Language Models with a Single GPU

tag: `FlexGen` | `ICML23` | `Stanford University`

paper link: [here](https://arxiv.org/pdf/2303.06865)

code link: [here](https://github.com/FMInference/FlexLLMGen)

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

tag: `ZeroInference` | `DeepSpeed` | `Microsoft`

blog link: [here](https://www.deepspeed.ai/2022/09/09/zero-inference.html)

code link: [here](https://github.com/microsoft/DeepSpeed/)

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

tag: `Orca` | `OSDI22` | `SNU`

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

tag: `BetterTransformer` | `PyTorch` | `Meta`

blog link: [here](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/)

homepage link: [here](https://huggingface.co/docs/optimum/bettertransformer/overview)

tutorial link: [here](../../tutorial/notebook/BetterTransformerDemo.ipynb)

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

tag: `DeepSpeed Inference`

blog link: [here](https://www.deepspeed.ai/2021/03/15/inference-kernel-optimization.html)

code link: [here](https://github.com/microsoft/DeepSpeed/)

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


## Survey

#### Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems

tag: `LLM Serving Survey` | `CMU`

paper link: [here](https://arxiv.org/pdf/2312.15234.pdf)

citation:

```bibtex
@misc{miao2023efficient,
      title={Towards Efficient Generative Large Language Model Serving: A Survey from Algorithms to Systems}, 
      author={Xupeng Miao and Gabriele Oliaro and Zhihao Zhang and Xinhao Cheng and Hongyi Jin and Tianqi Chen and Zhihao Jia},
      year={2023},
      eprint={2312.15234},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```