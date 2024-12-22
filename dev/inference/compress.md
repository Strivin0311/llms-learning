# Compression Strategies for LLMs Inference
*Here're some resources about Compression Strategies for LLMs Inference, especially for KV Cache memory compression and management*



#### ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference

tag: `ShadowKV` | `ByteDance` | `CMU`

paper link: [here](https://arxiv.org/pdf/2410.21465)

github link: [here](https://github.com/bytedance/ShadowKV/)

citation:

```bibtex
@misc{sun2024shadowkvkvcacheshadows,
      title={ShadowKV: KV Cache in Shadows for High-Throughput Long-Context LLM Inference}, 
      author={Hanshi Sun and Li-Wen Chang and Wenlei Bao and Size Zheng and Ningxin Zheng and Xin Liu and Harry Dong and Yuejie Chi and Beidi Chen},
      year={2024},
      eprint={2410.21465},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.21465}, 
}
```


#### PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling

tag: `PyramidKV` | `ByteDance` | `CMU`

paper link: [here](https://arxiv.org/pdf/2406.02069)

github link: [here](https://github.com/Zefan-Cai/KVCache-Factory)

citation:

```bibtex
@misc{cai2024pyramidkvdynamickvcache,
      title={PyramidKV: Dynamic KV Cache Compression based on Pyramidal Information Funneling}, 
      author={Zefan Cai and Yichi Zhang and Bofei Gao and Yuliang Liu and Tianyu Liu and Keming Lu and Wayne Xiong and Yue Dong and Baobao Chang and Junjie Hu and Wen Xiao},
      year={2024},
      eprint={2406.02069},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.02069}, 
}
```


#### Efficient memory management for large language model serving with pagedattention

tag: `Paged Attention` | `vLLM` | `SOSP23` | `UC Berkeley` | `Stanford University`

paper link: [here](https://dl.acm.org/doi/pdf/10.1145/3600006.3613165)

github link: [here](https://github.com/vllm-project/vllm)

citation:

```bibtex
@inproceedings{kwon2023efficient,
      author = {Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and Sheng, Ying and Zheng, Lianmin and Yu, Cody Hao and Gonzalez, Joseph and Zhang, Hao and Stoica, Ion},
      title = {Efficient Memory Management for Large Language Model Serving with PagedAttention},
      year = {2023},
      isbn = {9798400702297},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3600006.3613165},
      doi = {10.1145/3600006.3613165},
      pages = {611–626},
      numpages = {16},
      location = {Koblenz, Germany},
      series = {SOSP '23}
}
```