# Pipeline Parallelism for LLMs Training
*Here're some resources about Pipeline Parallelism for LLMs Training*


#### Hiding Communication Cost in Distributed LLM Training via Micro-batch Co-execution

tag: `DHelix` | `DNA PP` | `USTC`

paper link: [here](https://arxiv.org/pdf/2411.15871)

citation:

```bibtex
@misc{wang2024hidingcommunicationcostdistributed,
      title={Hiding Communication Cost in Distributed LLM Training via Micro-batch Co-execution}, 
      author={Haiquan Wang and Chaoyi Ruan and Jia He and Jiaqi Ruan and Chengjie Tang and Xiaosong Ma and Cheng Li},
      year={2024},
      eprint={2411.15871},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2411.15871}, 
}
```


#### Pipeline Parallelism with Controllable Memory

tag: `Controllable Memory` | `ZBPP` | `Sea AILab` | `NUS`

paper link: [here](https://arxiv.org/pdf/2405.15362)

code link: [here](https://github.com/sail-sg/zero-bubble-pipeline-parallelism)

citation:

```bibtex
@misc{qi2024pipelineparallelismcontrollablememory,
      title={Pipeline Parallelism with Controllable Memory}, 
      author={Penghui Qi and Xinyi Wan and Nyamdavaa Amar and Min Lin},
      year={2024},
      eprint={2405.15362},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.15362}, 
}
```

#### Zero Bubble (Almost) Pipeline Parallelism

tag: `ZBPP` | `ZB1P` | `ZB2P` | `ICLR24` | `Sea AILab`

paper link: [here](https://openreview.net/pdf?id=tuzTN0eIO5)

code link: [here](https://github.com/sail-sg/zero-bubble-pipeline-parallelism)

follow-up work: [here](https://arxiv.org/pdf/2405.15362)

citation:

```bibtex
@misc{qi2023zerobubblepipelineparallelism,
      title={Zero Bubble Pipeline Parallelism}, 
      author={Penghui Qi and Xinyi Wan and Guangxing Huang and Min Lin},
      year={2023},
      eprint={2401.10241},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2401.10241}, 
}
```


#### Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates

tag: `Oobleck` | `SOSP23` | `University of Michigan` | `Peking University`

paper link: [here](https://dl.acm.org/doi/pdf/10.1145/3600006.3613152)

code link: [here](https://github.com/SymbioticLab/Oobleck)

citation:

```bibtex
@inproceedings{oobleck-sosp23,
    title     = {Oobleck: Resilient Distributed Training of Large Models Using Pipeline Templates},
    author    = {Jang, Insu and Yang, Zhenning and Zhang, Zhen and Jin, Xin and Chowdhury, Mosharaf},
    booktitle = {ACM SIGOPS 29th Symposium of Operating Systems and Principles (SOSP '23)},
    year      = {2023},
}
```


#### BPipe: Memory-Balanced Pipeline Parallelism for Training Large Language Models

tag: `BPipe` | `Memory Balance` | `ICML23`

paper link: [here](https://proceedings.mlr.press/v202/kim23l/kim23l.pdf)

citation:

```bibtex
@inproceedings{kim2023bpipe,
  title = {{BP}ipe: Memory-Balanced Pipeline Parallelism for Training Large Language Models},
  author = {Kim, Taebum and Kim, Hyoungjoo and Yu, Gyeong-In and Chun, Byung-Gon},
  booktitle = {Proceedings of the 40th International Conference on Machine Learning},
  pages = {16639--16653},
  year = {2023},
  editor = {Krause, Andreas and Brunskill, Emma and Cho, Kyunghyun and Engelhardt, Barbara and Sabato, Sivan and Scarlett, Jonathan},
  volume = {202},
  series = {Proceedings of Machine Learning Research},
  month = {23--29 Jul},
  publisher = {PMLR},
  url = {https://proceedings.mlr.press/v202/kim23l.html},
}
```


#### Breadth-First Pipeline Parallelism

tag: `BFPP` | `MLSys23` | `ServiceNow Research`

paper link: [here](https://arxiv.org/pdf/2211.05953)

citation:

```bibtex
@misc{lamypoirier2023breadthfirstpipelineparallelism,
      title={Breadth-First Pipeline Parallelism}, 
      author={Joel Lamy-Poirier},
      year={2023},
      eprint={2211.05953},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2211.05953}, 
}
```

#### Chimera: Efficiently Training Large-Scale Neural Networks with Bidirectional Pipelines

tag: `Chimera` | `Bidirectional PP` | `SC21` | `ETHZ`

paper link: [here](https://arxiv.org/pdf/2107.06925)

citation:

```bibtex
@inproceedings{li2021chimera,
      author = {Li, Shigang and Hoefler, Torsten},
      title = {Chimera: efficiently training large-scale neural networks with bidirectional pipelines},
      year = {2021},
      isbn = {9781450384421},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3458817.3476145},
      doi = {10.1145/3458817.3476145},
      articleno = {27},
      numpages = {14},
      keywords = {pipeline parallelism, operator parallelism, model parallelism, distributed deep learning, data parallelism},
      location = {St. Louis, Missouri},
      series = {SC '21}
}
```


#### PipeTransformer: Automated Elastic Pipelining for Distributed Training of Large-scale Models

tag: `PipeTransformer` | `ICML21` | `Meta` | `USC`

paper link: [here](https://proceedings.mlr.press/v139/he21a/he21a.pdf)

slides link: [here](https://docs.google.com/presentation/d/1t6HWL33KIQo2as0nSHeBpXYtTBcy0nXCoLiKd0EashY/edit#slide=id.p)

code link: [here](https://github.com/Distributed-AI/PipeTransformer)

homepage link: [here](https://chaoyanghe.com/pipetransformer/)

citation:

```bibtex

@InProceedings{he2021pipetransformer,
  title = 	 {PipeTransformer: Automated Elastic Pipelining for Distributed Training of Large-scale Models},
  author =       {He, Chaoyang and Li, Shen and Soltanolkotabi, Mahdi and Avestimehr, Salman},
  booktitle = 	 {Proceedings of the 38th International Conference on Machine Learning},
  pages = 	 {4150--4159},
  year = 	 {2021},
  editor = 	 {Meila, Marina and Zhang, Tong},
  volume = 	 {139},
  series = 	 {Proceedings of Machine Learning Research},
  month = 	 {18--24 Jul},
  publisher =    {PMLR},
  pdf = 	 {http://proceedings.mlr.press/v139/he21a/he21a.pdf},
  url = 	 {https://proceedings.mlr.press/v139/he21a.html},
}
```


#### GEMS: GPU-Enabled Memory-Aware Model-Parallelism System for Distributed DNN Training

tag: `GEMS` | `SC20` | `OSU`

paper link: [here](https://ieeexplore.ieee.org/document/9355254)

citation:

```bibtex
@inproceedings{jain2020gems,
  author={Jain, Arpan and Awan, Ammar Ahmad and Aljuhani, Asmaa M. and Hashmi, Jahanzeb Maqbool and Anthony, Quentin G. and Subramoni, Hari and Panda, Dhableswar K. and Machiraju, Raghu and Parwani, Anil},
  booktitle={SC20: International Conference for High Performance Computing, Networking, Storage and Analysis}, 
  title={GEMS: GPU-Enabled Memory-Aware Model-Parallelism System for Distributed DNN Training}, 
  year={2020},
  volume={},
  number={},
  pages={1-15},
  keywords={Training;Histopathology;Computational modeling;High performance computing;Graphics processing units;Distributed databases;Data models;DNN;Model Parallelism;Keras;TensorFlow;Eager Execution;MPI},
  doi={10.1109/SC41405.2020.00049}
}
```


#### DAPPLE: A Pipelined Data Parallel Approach for Training Large Models

tag: `DAPPLE` | `PPoPP21` | `Alibaba Group`

paper link: [here](https://dl.acm.org/doi/pdf/10.1145/3437801.3441593)

citation:

```bibtex
@misc{fan2020dapple,
      title={DAPPLE: A Pipelined Data Parallel Approach for Training Large Models}, 
      author={Shiqing Fan and Yi Rong and Chen Meng and Zongyan Cao and Siyu Wang and Zhen Zheng and Chuan Wu and Guoping Long and Jun Yang and Lixue Xia and Lansong Diao and Xiaoyong Liu and Wei Lin},
      year={2020},
      eprint={2007.01045},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```


#### Memory-Efficient Pipeline-Parallel DNN Training

tag: `PipeDream` | `PipeDream-2BW` | `PipeDream-Flush` | `Double-Buffered Weight Updates` | `ICML21` | `Microsoft` | `Standford University`

paper link: [here](https://proceedings.mlr.press/v139/narayanan21a/narayanan21a.pdf)

citation:

```bibtex
@misc{narayanan2021memoryefficient,
      title={Memory-Efficient Pipeline-Parallel DNN Training}, 
      author={Deepak Narayanan and Amar Phanishayee and Kaiyu Shi and Xie Chen and Matei Zaharia},
      year={2021},
      eprint={2006.09503},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

#### torchgpipe: On-the-fly Pipeline Parallelism for Training Giant Models

tag: `torchgpipe` | `Kakao Brain`

paper link: [here](https://arxiv.org/pdf/2004.09910)

code link: [here](https://github.com/kakaobrain/torchgpipe)

citation:

```bibtex
@misc{kim2020torchgpipeontheflypipelineparallelism,
      title={torchgpipe: On-the-fly Pipeline Parallelism for Training Giant Models}, 
      author={Chiheon Kim and Heungsub Lee and Myungryong Jeong and Woonhyuk Baek and Boogeon Yoon and Ildoo Kim and Sungbin Lim and Sungwoong Kim},
      year={2020},
      eprint={2004.09910},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2004.09910}, 
}
```


#### PipeMare: Asynchronous Pipeline Parallel DNN Training

tag: `PipeMare` | `MLSys21` | `Stanford University`

paper link: [here](https://proceedings.mlsys.org/paper_files/paper/2021/file/9412531719be7ccf755c4ff98d0969dc-Paper.pdf)

citation:

```bibtex
@misc{yang2020pipemareasynchronouspipelineparallel,
      title={PipeMare: Asynchronous Pipeline Parallel DNN Training}, 
      author={Bowen Yang and Jian Zhang and Jonathan Li and Christopher Ré and Christopher R. Aberger and Christopher De Sa},
      year={2020},
      eprint={1910.05124},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/1910.05124}, 
}
```

#### Pipe-Torch: Pipeline-Based Distributed Deep Learning in a GPU Cluster with Heterogeneous Networking

tag: `Pipe-Torch` | `CBD19` | `SEU`

paper link: [here](https://ieeexplore.ieee.org/document/8916305)

citation:

```bibtex
@inproceedings{zhan2019pipetorch,
  author={Zhan, Jun and Zhang, Jinghui},
  booktitle={2019 Seventh International Conference on Advanced Cloud and Big Data (CBD)}, 
  title={Pipe-Torch: Pipeline-Based Distributed Deep Learning in a GPU Cluster with Heterogeneous Networking}, 
  year={2019},
  volume={},
  number={},
  pages={55-60},
  keywords={Graphics processing units;Training;Parallel processing;Computational modeling;Data models;Pipelines;Load modeling;Distributed deep learning;Pipeline-hybrid parallelism;Heterogeneous network environment},
  doi={10.1109/CBD.2019.00020}
}
```

#### GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism

tag: `GPipe` | `NIPS19` | `Google`

paper link: [here](https://papers.nips.cc/paper_files/paper/2019/file/093f65e080a295f8076b1c5722a46aa2-Paper.pdf)

citation:

```bibtex
@misc{huang2019gpipe,
      title={GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism}, 
      author={Yanping Huang and Youlong Cheng and Ankur Bapna and Orhan Firat and Mia Xu Chen and Dehao Chen and HyoukJoong Lee and Jiquan Ngiam and Quoc V. Le and Yonghui Wu and Zhifeng Chen},
      year={2019},
      eprint={1811.06965},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```


#### PipeDream: Fast and Efficient Pipeline Parallel DNN Training

tag: `PipeDream` | `SOSP19` | `Microsoft` | `CMU` | `Stanford University`

paper link: [here](https://www.pdl.cmu.edu/PDL-FTP/BigLearning/sosp19-final271.pdf)

citation:

```bibtex
@inproceedings{harlap2018pipedream,
      author = {Narayanan, Deepak and Harlap, Aaron and Phanishayee, Amar and Seshadri, Vivek and Devanur, Nikhil R. and Ganger, Gregory R. and Gibbons, Phillip B. and Zaharia, Matei},
      title = {PipeDream: generalized pipeline parallelism for DNN training},
      year = {2019},
      isbn = {9781450368735},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3341301.3359646},
      doi = {10.1145/3341301.3359646},
      booktitle = {Proceedings of the 27th ACM Symposium on Operating Systems Principles},
      pages = {1–15},
      numpages = {15},
      location = {Huntsville, Ontario, Canada},
      series = {SOSP '19}
}
```


