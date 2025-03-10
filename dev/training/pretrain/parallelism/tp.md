# Tensor Parallelism for LLMs Training
*Here're some resources about Tensor Parallelism for LLMs Training*
*Note that the "sequence parallelism" usually refers to one attached parallelism strategy along with tensor parallelism*



#### Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts

tag: `Comet` | `FLUX` | `Async TP` | `MoE` | `ByteDance`

paper link: [here](https://arxiv.org/pdf/2502.19811)

code link: [here](https://github.com/bytedance/flux)

citation:

```bibtex
@misc{zhang2025cometfinegrainedcomputationcommunicationoverlapping,
      title={Comet: Fine-grained Computation-communication Overlapping for Mixture-of-Experts}, 
      author={Shulai Zhang and Ningxin Zheng and Haibin Lin and Ziheng Jiang and Wenlei Bao and Chengquan Jiang and Qi Hou and Weihao Cui and Size Zheng and Li-Wen Chang and Quan Chen and Xin Liu},
      year={2025},
      eprint={2502.19811},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2502.19811}, 
}
```


#### FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion

tag: `FLUX` | `Async TP` | `NVShmem` | `ByteDance`

paper link: [here](https://arxiv.org/pdf/2406.06858)

code link: [here](https://github.com/bytedance/flux)

follow-up work: [here](https://arxiv.org/pdf/2502.19811)

citation:

```bibtex
@misc{chang2024fluxfastsoftwarebasedcommunication,
      title={FLUX: Fast Software-based Communication Overlap On GPUs Through Kernel Fusion}, 
      author={Li-Wen Chang and Wenlei Bao and Qi Hou and Chengquan Jiang and Ningxin Zheng and Yinmin Zhong and Xuanrun Zhang and Zuquan Song and Ziheng Jiang and Haibin Lin and Xin Jin and Xin Liu},
      year={2024},
      eprint={2406.06858},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2406.06858}, 
}
```


#### Stream-K- Work-centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU

tag: `Stream-K` | `PPoPP23` | `Nvidia`

paper link: [here](https://dl.acm.org/doi/pdf/10.1145/3572848.3577479)

citation:

```bibtex
@inproceedings{10.1145/3572848.3577479,
      author = {Osama, Muhammad and Merrill, Duane and Cecka, Cris and Garland, Michael and Owens, John D.},
      title = {Stream-K: Work-Centric Parallel Decomposition for Dense Matrix-Matrix Multiplication on the GPU},
      year = {2023},
      isbn = {9798400700156},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3572848.3577479},
      doi = {10.1145/3572848.3577479},
      booktitle = {Proceedings of the 28th ACM SIGPLAN Annual Symposium on Principles and Practice of Parallel Programming},
      pages = {429–431},
      numpages = {3},
      keywords = {GPU, load-balancing, matrix-multiplication},
      location = {Montreal, QC, Canada},
      series = {PPoPP '23}
}
```


#### Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models

tag: `Async TP` | `ASPLOS23`| `Google`

paper link: [here](https://dl.acm.org/doi/pdf/10.1145/3567955.3567959)

citation:

```bibtex
@inproceedings{wang2023overlap,
      author = {Wang, Shibo and Wei, Jinliang and Sabne, Amit and Davis, Andy and Ilbeyi, Berkin and Hechtman, Blake and Chen, Dehao and Murthy, Karthik Srinivasa and Maggioni, Marcello and Zhang, Qiao and Kumar, Sameer and Guo, Tongfei and Xu, Yuanzhong and Zhou, Zongwei},
      title = {Overlap Communication with Dependent Computation via Decomposition in Large Deep Learning Models},
      year = {2022},
      isbn = {9781450399159},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3567955.3567959},
      doi = {10.1145/3567955.3567959},
      booktitle = {Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 1},
      pages = {93–106},
      numpages = {14},
      keywords = {Collective communication hiding, Compiler optimization, Large scale machine learning},
      location = {Vancouver, BC, Canada},
      series = {ASPLOS 2023}
}
```


#### Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads

tag: `CoCoNeT` | `Async TP` | `ASPLOS22`| `Microsoft`

paper link: [here](https://dl.acm.org/doi/pdf/10.1145/3503222.3507778)

citation:

```bibtex
@inproceedings{10.1145/3503222.3507778,
      author = {Jangda, Abhinav and Huang, Jun and Liu, Guodong and Sabet, Amir Hossein Nodehi and Maleki, Saeed and Miao, Youshan and Musuvathi, Madanlal and Mytkowicz, Todd and Saarikivi, Olli},
      title = {Breaking the computation and communication abstraction barrier in distributed machine learning workloads},
      year = {2022},
      isbn = {9781450392051},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3503222.3507778},
      doi = {10.1145/3503222.3507778},
      pages = {402–416},
      numpages = {15},
      keywords = {CUDA, Code Generation, Collective Communication, Compiler Optimizations, Distributed Machine Learning, MPI},
      location = {Lausanne, Switzerland},
      series = {ASPLOS '22}
}
```


#### Reducing Activation Recomputation in Large Transformer Models

tag: `SP` | `TSP` | `Sequence Parallelism` | `MLSys23` | `Nvidia`

paper link: [here](https://proceedings.mlsys.org/paper_files/paper/2023/file/80083951326cf5b35e5100260d64ed81-Paper-mlsys2023.pdf)

code link: [here](https://github.com/NVIDIA/Megatron-LM)

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


#### Maximizing Parallelism in Distributed Training for Huge Neural Networks

tag: `3-D TP` | `3-D Parallel Matrix Multiplication` | `NUS`

paper link: [here](https://arxiv.org/pdf/2105.14450)

citation:

```bibtex
@misc{bian2021maximizing,
      title={Maximizing Parallelism in Distributed Training for Huge Neural Networks}, 
      author={Zhengda Bian and Qifan Xu and Boxiang Wang and Yang You},
      year={2021},
      eprint={2105.14450},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```

#### Tesseract: Parallelize the Tensor Parallelism Efficiently

tag: `Tesseract` | `2.5-D TP` | `SUMMA` | `ICPP22` | `NUS`

paper link: [here](https://arxiv.org/pdf/2105.14500)

citation:

```bibtex
@inproceedings{wang2022tesseract,
      author = {Wang, Boxiang and Xu, Qifan and Bian, Zhengda and You, Yang},
      title = {Tesseract: Parallelize the Tensor Parallelism Efficiently},
      year = {2023},
      isbn = {9781450397339},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3545008.3545087},
      doi = {10.1145/3545008.3545087},
      articleno = {12},
      numpages = {11},
      keywords = {Parallelism, Machine Learning, MLsys},
      location = {Bordeaux, France},
      series = {ICPP '22}
}
```


#### An Efficient 2D Method for Training Super-Large Deep Learning Models

tag: `2-D TP` | `SUMMA` | `IPDPS23` | `UCLA`

paper link: [here](https://arxiv.org/pdf/2104.05343)

citation:

```bibtex
@misc{xu2021efficient,
      title={An Efficient 2D Method for Training Super-Large Deep Learning Models}, 
      author={Qifan Xu and Shenggui Li and Chaoyu Gong and Yang You},
      year={2021},
      eprint={2104.05343},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

#### Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism

tag: `Megatron-LM` | `Column Linear` | `Row Linear` | `Nvidia`

paper link: [here](https://arxiv.org/pdf/1909.08053.pdf)

code link: [here](https://github.com/NVIDIA/Megatron-LM)

follow-up work: [here](https://arxiv.org/pdf/2104.04473.pdf)

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

#### Local SGD Converges Fast and Communicates Little

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


#### Parallel SGD: When does averaging help?

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


