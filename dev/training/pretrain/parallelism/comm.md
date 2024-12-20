# Distributed Communication for LLMs Training
*Here're some resources about Distributed Communication for LLMs Training*


## Method


#### Starburst: A Cost-aware Scheduler for Hybrid Cloud

tag: `Starburst` | `ATC24` | `UCB`

paper link: [here](https://www.usenix.org/system/files/atc24-luo.pdf)

citation:

```bibtex
@inproceedings{luo2024starburst,
  title={Starburst: A Cost-aware Scheduler for Hybrid Cloud},
  author={Luo, Michael and Zhuang, Siyuan and Vengadesan, Suryaprakash and Bhardwaj, Romil and Chang, Justin and Friedman, Eric and Shenker, Scott and Stoica, Ion},
  booktitle={2024 USENIX Annual Technical Conference (USENIX ATC 24)},
  pages={37--57},
  year={2024}
}
```


#### OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training

tag: `OpenDiLoCo` | `Prime Intellect`

paper link: [here](https://arxiv.org/pdf/2407.07852)

github link: [here](https://github.com/PrimeIntellect-ai/OpenDiLoCo)

citation:

```bibtex
@misc{jaghouar2024opendilocoopensourceframeworkglobally,
      title={OpenDiLoCo: An Open-Source Framework for Globally Distributed Low-Communication Training}, 
      author={Sami Jaghouar and Jack Min Ong and Johannes Hagemann},
      year={2024},
      eprint={2407.07852},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.07852}, 
}
```


#### DiLoCo: Distributed Low-Communication Training of Language Models

tag: `DiLoCo` | `Google DeepMind`

paper link: [here](https://arxiv.org/pdf/2311.08105)

follow-up work: [here](https://arxiv.org/pdf/2407.07852)

citation:

```bibtex
@misc{douillard2023diloco,
      title={DiLoCo: Distributed Low-Communication Training of Language Models}, 
      author={Arthur Douillard and Qixuan Feng and Andrei A. Rusu and Rachita Chhaparia and Yani Donchev and Adhiguna Kuncoro and Marc'Aurelio Ranzato and Arthur Szlam and Jiajun Shen},
      year={2023},
      eprint={2311.08105},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### ZeRO++: Extremely Efficient Collective Communication for Giant Model Training

tag: `ZeRO++` | `DeepSpeed` | `ICLR24` | `Microsoft`

paper link: [here](https://arxiv.org/pdf/2306.10209)

blog link: [here](https://www.microsoft.com/en-us/research/blog/deepspeed-zero-a-leap-in-speed-for-llm-and-chat-model-training-with-4x-less-communication/)

slides link: [here](https://nips.cc/media/neurips-2023/Slides/84328_0BRW6hm.pdf)

github link: [here](https://github.com/microsoft/DeepSpeed)

citation:

```bibtex
@misc{wang2023zeroextremelyefficientcollective,
      title={ZeRO++: Extremely Efficient Collective Communication for Giant Model Training}, 
      author={Guanhua Wang and Heyang Qin and Sam Ade Jacobs and Connor Holmes and Samyam Rajbhandari and Olatunji Ruwase and Feng Yan and Lei Yang and Yuxiong He},
      year={2023},
      eprint={2306.10209},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2306.10209}, 
}
```


#### MSCCLang: Microsoft Collective Communication Language

tag: `MSCCLang` | `ASPLOS23` | `Microsoft`

paper link: [here](https://dl.acm.org/doi/pdf/10.1145/3575693.3575724)

github link: [here](https://github.com/microsoft/msccl)

citation:

```bibtex
@inproceedings{10.1145/3575693.3575724,
      author = {Cowan, Meghan and Maleki, Saeed and Musuvathi, Madanlal and Saarikivi, Olli and Xiong, Yifan},
      title = {MSCCLang: Microsoft Collective Communication Language},
      year = {2023},
      isbn = {9781450399166},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3575693.3575724},
      doi = {10.1145/3575693.3575724},
      booktitle = {Proceedings of the 28th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2},
      pages = {502–514},
      numpages = {13},
      keywords = {Collective Communication, Compilers, GPU},
      location = {Vancouver, BC, Canada},
      series = {ASPLOS 2023}
}
```

#### TACCL: Guiding Collective Algorithm Synthesis using Communication Sketches

tag: `TACCL` | `OSDI23` | `Microsoft`

paper link: [here](https://www.usenix.org/system/files/nsdi23-shah.pdf)

citation:

```bibtex
@inproceedings {285084,
      author = {Aashaka Shah and Vijay Chidambaram and Meghan Cowan and Saeed Maleki and Madan Musuvathi and Todd Mytkowicz and Jacob Nelson and Olli Saarikivi and Rachee Singh},
      title = {{TACCL}: Guiding Collective Algorithm Synthesis using Communication Sketches},
      booktitle = {20th USENIX Symposium on Networked Systems Design and Implementation (NSDI 23)},
      year = {2023},
      isbn = {978-1-939133-33-5},
      address = {Boston, MA},
      pages = {593--612},
      url = {https://www.usenix.org/conference/nsdi23/presentation/shah},
      publisher = {USENIX Association},
      month = apr
}
```


#### On Optimizing the Communication of Model Parallelism

tag: `AlpaComm` | `MBZUAI` | `CMU` | `Tsinghua University` | `UCB`

paper link: [here](https://arxiv.org/pdf/2211.05322)

citation:

```bibtex
@misc{zhuang2024optimizingcommunicationmodelparallelism,
      title={On Optimizing the Communication of Model Parallelism}, 
      author={Yonghao Zhuang and Hexu Zhao and Lianmin Zheng and Zhuohan Li and Eric P. Xing and Qirong Ho and Joseph E. Gonzalez and Ion Stoica and Hao Zhang},
      year={2024},
      eprint={2211.05322},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2211.05322}, 
}
```

#### Enabling Compute-Communication Overlap in Distributed Deep Learning Training Platforms

tag: `ACE` | `ISCA21`

paper link: [here](https://arxiv.org/pdf/2211.05322)

citation:

```bibtex
@inproceedings{rashidi2021enabling,
      author = {Rashidi, Saeed and Denton, Matthew and Sridharan, Srinivas and Srinivasan, Sudarshan and Suresh, Amoghavarsha and Nie, Jade and Krishna, Tushar},
      title = {Enabling compute-communication overlap in distributed deep learning training platforms},
      year = {2021},
      isbn = {9781450390866},
      publisher = {IEEE Press},
      url = {https://doi.org/10.1109/ISCA52012.2021.00049},
      doi = {10.1109/ISCA52012.2021.00049},
      pages = {540–553},
      numpages = {14},
      keywords = {deep learning training, communication accelerator, collective communication, accelerator fabric},
      location = {Virtual Event, Spain},
      series = {ISCA '21}
}
```


#### Breaking the Computation and Communication Abstraction Barrier in Distributed Machine Learning Workloads

tag: `CoCoNet` | `ASPLOS22`| `Microsoft`

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


#### Efficient sparse collective communication and its application to accelerate distributed deep learning

tag: `OmniReduce` | `SIGCOMM21` | `NUDT`

paper link: [here](https://dl.acm.org/doi/pdf/10.1145/3452296.3472904)

citation:

```bibtex
@inproceedings{fei2021efficient,
      author = {Fei, Jiawei and Ho, Chen-Yu and Sahu, Atal N. and Canini, Marco and Sapio, Amedeo},
      title = {Efficient sparse collective communication and its application to accelerate distributed deep learning},
      year = {2021},
      isbn = {9781450383837},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3452296.3472904},
      doi = {10.1145/3452296.3472904},
      booktitle = {Proceedings of the 2021 ACM SIGCOMM 2021 Conference},
      pages = {676–691},
      numpages = {16},
      keywords = {distributed training, deep learning},
      location = {Virtual Event, USA},
      series = {SIGCOMM '21}
}
```


#### Plink: Discovering and Exploiting Datacenter Network Locality for Efficient Cloud-Based Distributed Training

tag: `Plink` | `MLSys20` | `Microsoft` | `University of Washington`

paper link: [here](https://proceedings.mlsys.org/paper_files/paper/2020/file/eca986d585a03890a412587a2f5ccb43-Paper.pdf)

citation:

```bibtex
@inproceedings{luo2020plink,
  title={Plink: Discovering and Exploiting Datacenter Network Locality for Efficient Cloud-based Distributed Training},
  author={Luo, Liang and West, Peter and Krishnamurthy, Arvind and Ceze, Luis and Nelson, Jacob},
  booktitle={Proceedings of the 3rd MLSys Conference},
  pages={455--469},
  year={2020},
  organization={PMLR}
}
```


#### IncBricks: Toward In-Network Computation with an In-Network Cache

tag: `IncBricks` | `ASPLOS17` | `University of Washington`

paper link: [here](https://dl.acm.org/doi/pdf/10.1145/3037697.3037731)

citation:

```bibtex
@inproceedings{liu2017incbricks,
      author = {Liu, Ming and Luo, Liang and Nelson, Jacob and Ceze, Luis and Krishnamurthy, Arvind and Atreya, Kishore},
      title = {IncBricks: Toward In-Network Computation with an In-Network Cache},
      year = {2017},
      isbn = {9781450344654},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3037697.3037731},
      doi = {10.1145/3037697.3037731},
      booktitle = {Proceedings of the Twenty-Second International Conference on Architectural Support for Programming Languages and Operating Systems},
      pages = {795–809},
      numpages = {15},
      keywords = {in-network caching, programmable network devices},
      location = {Xi'an, China},
      series = {ASPLOS '17}
}
```


#### ClickNP: Highly Flexible and High Performance Network Processing with Reconfigurable Hardware

tag: `ClickNP` | `SIGCOMM16` | `Microsoft` | `USTC`

paper link: [here](https://dl.acm.org/doi/pdf/10.1145/2934872.2934897)

citation:

```bibtex
@inproceedings{li2016clicknp,
  title={ClickNP: Highly Flexible and High Performance Network Processing with Reconfigurable Hardware},
  author={Li, Bojie and Tan, Kun and Xu, Ningyi and Luo, Layong and Xiong, Yongqiang and Peng, Yanqing and Luo, Renqian and Cheng, Peng and Chen, Enhong},
  booktitle={Proceedings of the 2016 ACM SIGCOMM Conference},
  pages={205--218},
  year={2016},
  organization={ACM}
}
```


## Survey


#### Communication-Efficient Distributed Deep Learning: A Comprehensive Survey

tag: `Communication-Efficient` | `Distributed DL` | `Survey` | `HKBU`

paper link: [here](https://arxiv.org/pdf/2003.06307)

citation:

```bibtex
@misc{tang2023communicationefficientdistributeddeeplearning,
      title={Communication-Efficient Distributed Deep Learning: A Comprehensive Survey}, 
      author={Zhenheng Tang and Shaohuai Shi and Wei Wang and Bo Li and Xiaowen Chu},
      year={2023},
      eprint={2003.06307},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2003.06307}, 
}
```