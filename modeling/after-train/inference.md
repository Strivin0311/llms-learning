# Inference on LLMs
*Here're some resources about Inference on LLMs*


#### Fast Inference of Mixture-of-Experts Language Models with Offloading [`READ`]

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

#### PowerInfer: Fast Large Language Model Serving with a Consumer-grade GPU [`UNREAD`]

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

#### LLM in a flash: Efficient Large Language Model Inference with Limited Memory [`UNREAD`]

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

#### S-LoRA: Serving Thousands of Concurrent LoRA Adapters [`READ`]

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

#### DeepSpeed-FastGen: High-throughput Text Generation for LLMs via MII and DeepSpeed-Inference [`READ`]

blog link: [here](https://github.com/microsoft/DeepSpeed/tree/master/blogs/deepspeed-fastgen)

github link: [here](https://github.com/microsoft/DeepSpeed-MII)

tutorial links:

|tutorial name|public date|main-lib version|notebook link|
|-|-|-|-|
|tutorial_deepspeed_infer|2024.01|deepspeed=0.12.6, deepspeed-mii=0.1.3, transformers=4.36.2|[here](../notebooks/tutorial_deepspeed_infer.ipynb)|

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


#### AirLLM: inference 70B LLM with 4GB single GPU [`READ`]

github link: [here](https://github.com/lyogavin/Anima/tree/main/air_llm)

tutorial links:

|tutorial name|public date|main-lib version|notebook link|
|-|-|-|-|
|tutorial_airllm|2024.01|airllm=2.8.3|[here](../notebooks/tutorial_airllm.ipynb)|

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


#### High-throughput generative inference of large language models with a single gpu [`UNREAD`]

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

#### ZeRO-Inference: Democratizing massive model inference [`READ`]

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


#### A BetterTransformer for Fast Transformer Inference [`READ`]

blog link: [here](https://pytorch.org/blog/a-better-transformer-for-fast-transformer-encoder-inference/)

homepage link: [here](https://huggingface.co/docs/optimum/bettertransformer/overview)

tutorial link: [here](../notebooks/BetterTransformerDemo.ipynb)

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

#### DeepSpeed Inference: Multi-GPU inference with customized inference kernels and quantization support [`READ`]

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


#### Fast transformer decoding: One write-head is all you need [`READ`]

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
    
    