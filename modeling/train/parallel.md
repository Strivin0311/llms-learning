# Parallelism Optimization
*Here're some resources about Parallelism optimization strategies in LLMs training*


#### ZeRO++: Extremely Efficient Collective Communication for Giant Model Training [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2306.10209.pdf)

blog link: [here](https://www.microsoft.com/en-us/research/blog/deepspeed-zero-a-leap-in-speed-for-llm-and-chat-model-training-with-4x-less-communication/)

citation:
```bibtex
@misc{wang2023zero,
      title={ZeRO++: Extremely Efficient Collective Communication for Giant Model Training}, 
      author={Guanhua Wang and Heyang Qin and Sam Ade Jacobs and Connor Holmes and Samyam Rajbhandari and Olatunji Ruwase and Feng Yan and Lei Yang and Yuxiong He},
      year={2023},
      eprint={2306.10209},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```


#### Pytorch FSDP: experiences on scaling fully sharded data parallel [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2304.11277)

blog link: [here](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)

docs link: [here](https://pytorch.org/docs/stable/fsdp.html)

citation: 
```bibtex
@article{zhao2023pytorch,
  title={Pytorch FSDP: experiences on scaling fully sharded data parallel},
  author={Zhao, Yanli and Gu, Andrew and Varma, Rohan and Luo, Liang and Huang, Chien-Chin and Xu, Min and Wright, Less and Shojanazeri, Hamid and Ott, Myle and Shleifer, Sam and others},
  journal={arXiv preprint arXiv:2304.11277},
  year={2023}
}
```

#### Reducing Activation Recomputation in Large Transformer Models (Sequence Parallelism) [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2205.05198.pdf)

citation:
```bibtex
@misc{korthikanti2022reducing,
      title={Reducing Activation Recomputation in Large Transformer Models}, 
      author={Vijay Korthikanti and Jared Casper and Sangkug Lym and Lawrence McAfee and Michael Andersch and Mohammad Shoeybi and Bryan Catanzaro},
      year={2022},
      eprint={2205.05198},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2104.07857.pdf)

blog link: [here](https://www.microsoft.com/en-us/research/blog/zero-infinity-and-deepspeed-unlocking-unprecedented-model-scale-for-deep-learning-training/)

citation:
```bibtex
@misc{rajbhandari2021zeroinfinity,
      title={ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning}, 
      author={Samyam Rajbhandari and Olatunji Ruwase and Jeff Rasley and Shaden Smith and Yuxiong He},
      year={2021},
      eprint={2104.07857},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```


#### Colossal-ai: A unified deep learning system for large-scale parallel training (Auto Parallel) [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2110.14883)

github link: [here](https://github.com/hpcaitech/ColossalAI)

citation: 
```bibtex
@inproceedings{li2023colossal,
  title={Colossal-ai: A unified deep learning system for large-scale parallel training},
  author={Li, Shenggui and Liu, Hongxin and Bian, Zhengda and Fang, Jiarui and Huang, Haichen and Liu, Yuliang and Wang, Boxiang and You, Yang},
  booktitle={Proceedings of the 52nd International Conference on Parallel Processing},
  pages={766--775},
  year={2023}
}
```


#### Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (Pipeline Parallelism) [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2104.04473.pdf)

citation:
```bibtex
@misc{narayanan2021efficient,
      title={Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM}, 
      author={Deepak Narayanan and Mohammad Shoeybi and Jared Casper and Patrick LeGresley and Mostofa Patwary and Vijay Anand Korthikanti and Dmitri Vainbrand and Prethvi Kashinkunti and Julie Bernauer and Bryan Catanzaro and Amar Phanishayee and Matei Zaharia},
      year={2021},
      eprint={2104.04473},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### ZeRO-Offload: Democratizing Billion-Scale Model Training [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2101.06840.pdf)

blog link: [here](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)

citation:
```bibtex
@misc{ren2021zerooffload,
      title={ZeRO-Offload: Democratizing Billion-Scale Model Training}, 
      author={Jie Ren and Samyam Rajbhandari and Reza Yazdani Aminabadi and Olatunji Ruwase and Shuangyan Yang and Minjia Zhang and Dong Li and Yuxiong He},
      year={2021},
      eprint={2101.06840},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```

#### DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters [`READ`]

paper link: [here](https://dl.acm.org/doi/10.1145/3394486.3406703)

github link: [here](https://github.com/microsoft/DeepSpeed)

docs link: [here](https://deepspeed.readthedocs.io/en/latest/index.html)

citation:
```bibtex
@inproceedings{rasley2020deepspeed,
  title={Deepspeed: System optimizations enable training deep learning models with over 100 billion parameters},
  author={Rasley, Jeff and Rajbhandari, Samyam and Ruwase, Olatunji and He, Yuxiong},
  booktitle={Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery \& Data Mining},
  pages={3505--3506},
  year={2020}
}
```


#### Automatic Cross-Replica Sharding of Weight Update in Data-Parallel Training [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2004.13336.pdf)

citation:
```bibtex
@misc{xu2020automatic,
      title={Automatic Cross-Replica Sharding of Weight Update in Data-Parallel Training}, 
      author={Yuanzhong Xu and HyoukJoong Lee and Dehao Chen and Hongjun Choi and Blake Hechtman and Shibo Wang},
      year={2020},
      eprint={2004.13336},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```


#### ZeRO: Memory Optimizations Toward Training Trillion Parameter Models [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/1910.02054.pdf)

blog link: [here](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

citation:
```bibtex
@misc{rajbhandari2020zero,
      title={ZeRO: Memory Optimizations Toward Training Trillion Parameter Models}, 
      author={Samyam Rajbhandari and Jeff Rasley and Olatunji Ruwase and Yuxiong He},
      year={2020},
      eprint={1910.02054},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/1909.08053.pdf)

github link: [here](https://github.com/NVIDIA/Megatron-LM)

citation:
```bibtex
@misc{shoeybi2020megatronlm,
      title={Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism}, 
      author={Mohammad Shoeybi and Mostofa Patwary and Raul Puri and Patrick LeGresley and Jared Casper and Bryan Catanzaro},
      year={2020},
      eprint={1909.08053},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

#### Local SGD Converges Fast and Communicates Little [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/1805.09767.pdf)

citation:
```bibtex
@misc{stich2019local,
      title={Local SGD Converges Fast and Communicates Little}, 
      author={Sebastian U. Stich},
      year={2019},
      eprint={1805.09767},
      archivePrefix={arXiv},
      primaryClass={math.OC}
}
```


#### Parallel SGD: When does averaging help? [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/1606.07365.pdf)

citation:
```bibtex
@misc{zhang2016parallel,
      title={Parallel SGD: When does averaging help?}, 
      author={Jian Zhang and Christopher De Sa and Ioannis Mitliagkas and Christopher Ré},
      year={2016},
      eprint={1606.07365},
      archivePrefix={arXiv},
      primaryClass={stat.ML}
}
```