# Checkpointing Systems for LLMs Training
*Here're some resources about checkpointing systems for LLMs Training*


#### ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development

tag: `ByteCheckpoint` | `ByteDance`

paper link: [here](https://arxiv.org/pdf/2407.20143)

citation:

```bibtex
@misc{wan2024bytecheckpointunifiedcheckpointinglarge,
      title={ByteCheckpoint: A Unified Checkpointing System for Large Foundation Model Development}, 
      author={Borui Wan and Mingji Han and Yiyao Sheng and Yanghua Peng and Haibin Lin and Mofan Zhang and Zhichao Lai and Menghan Yu and Junda Zhang and Zuquan Song and Xin Liu and Chuan Wu},
      year={2024},
      eprint={2407.20143},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2407.20143}, 
}
```


#### Flash Checkpoint to Recover Large Model Training From Failure in Seconds

tag: `Flash Checkpoint` | `DLrover` | `AntGroup`

blog link: [here](https://github.com/intelligent-machine-learning/dlrover/blob/master/docs/blogs/flash_checkpoint.md)

github link: [here](https://github.com/intelligent-machine-learning/dlrover)

citation:

```bibtex
@misc{zhang2022flash,
      title={Flash Checkpoint to Recover Large Model Training From Failure in Seconds}, 
      author={Qinlong Wang},
      year={2024},
      howpublished={https://github.com/intelligent-machine-learning/dlrover/blob/master/docs/blogs/flash_checkpoint.md},
}
```


#### NVIDIA Megatron Distributed Checkpointing

tag: `Megatron DistCheckpointing` | `NVIDIA`

docs link: [here](https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/dist_checkpointing.html)

citation:

```bibtex
@manual{megatrondistckpt2024nvidia,
  title = {Distributed Checkpointing API Guide},
  author= {NVIDIA},
  year  = {2024},
  url   = {https://docs.nvidia.com/megatron-core/developer-guide/latest/api-guide/dist_checkpointing.html},
  note  = {Accessed: 2024-10-27},
  organization = {NVIDIA Corporation}
}

```


#### Just-In-Time Checkpointing: Low Cost Error Recovery from Deep Learning Training Failures

tag: `JIT Checkpointing` | `EuroSys24` | `Microsoft`

paper link: [here](https://dl.acm.org/doi/pdf/10.1145/3627703.3650085)

citation:

```bibtex
@inproceedings{justintime2024gupta,
    author = {Gupta, Tanmaey and Krishnan, Sanjeev and Kumar, Rituraj and Vijeev, Abhishek and Gulavani, Bhargav and Kwatra, Nipun and Ramjee, Ramachandran and Sivathanu, Muthian},
    title = {Just-In-Time Checkpointing: Low Cost Error Recovery from Deep Learning Training Failures},
    year = {2024},
    isbn = {9798400704376},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3627703.3650085},
    doi = {10.1145/3627703.3650085},
    pages = {1110–1125},
    numpages = {16},
    keywords = {Large Scale DNN Training Reliability, Reliable Distributed Systems, Systems for Machine Learning},
    location = {Athens, Greece},
    series = {EuroSys '24}
}
```


#### Getting Started with Distributed Checkpoint (DCP)

tag: `DCP` | `PyTorch`

docs link: [here](https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html)

citation:

```bibtex
@misc{zhang2024distributedcheckpoint,
  author = {Iris Zhang and Rodrigo Kumpera and Chien-Chin Huang and Lucas Pasqualin},
  title = {Getting Started with Distributed Checkpoint (DCP)},
  year = {2024},
  howpublished = {\url{https://pytorch.org/tutorials/recipes/distributed_checkpoint_recipe.html}},
}
```


#### GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints

tag: `GEMINI` | `SOSP23` | `AWS`

paper link: [here](https://dl.acm.org/doi/pdf/10.1145/3600006.3613145)

citation:

```bibtex
@inproceedings{gemini2023wang,
    author = {Wang, Zhuang and Jia, Zhen and Zheng, Shuai and Zhang, Zhen and Fu, Xinwei and Ng, T. S. Eugene and Wang, Yida},
    title = {GEMINI: Fast Failure Recovery in Distributed Training with In-Memory Checkpoints},
    year = {2023},
    isbn = {9798400702297},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3600006.3613145},
    doi = {10.1145/3600006.3613145},
    pages = {364–381},
    numpages = {18},
    keywords = {in-memory checkpoint, fault tolerance, distributed training},
    location = {Koblenz, Germany},
    series = {SOSP '23}
}
```


#### Check-N-Run: a Checkpointing System for Training Deep Learning Recommendation Models

tag: `Check-N-Run` | `NSDI22` | `Meta`

paper link: [here](https://www.usenix.org/system/files/nsdi22-paper-eisenman.pdf)

slides link: [here](https://www.usenix.org/system/files/nsdi22_slides_eisenman.pdf)

citation:

```bibtex
@inproceedings {checknrun2022eisenman,
    author = {Assaf Eisenman and Kiran Kumar Matam and Steven Ingram and Dheevatsa Mudigere and Raghuraman Krishnamoorthi and Krishnakumar Nair and Misha Smelyanskiy and Murali Annavaram},
    title = {{Check-N-Run}: a Checkpointing System for Training Deep Learning Recommendation Models},
    booktitle = {19th USENIX Symposium on Networked Systems Design and Implementation (NSDI 22)},
    year = {2022},
    isbn = {978-1-939133-27-4},
    address = {Renton, WA},
    pages = {929--943},
    url = {https://www.usenix.org/conference/nsdi22/presentation/eisenman},
    publisher = {USENIX Association},
    month = apr
}
```


#### CheckFreq: Frequent, Fine-Grained DNN Checkpointing

tag: `CheckFreq` | `FAST21` | `Microsoft`

paper link: [here](https://www.usenix.org/system/files/fast21-mohan.pdf)

slides link: [here](https://www.usenix.org/sites/default/files/conference/protected-files/fast21_slides_mohan.pdf)

citation:

```bibtex
@inproceedings {checkfreq2021mohan,
    author = {Jayashree Mohan and Amar Phanishayee and Vijay Chidambaram},
    title = {{CheckFreq}: Frequent, {Fine-Grained} {DNN} Checkpointing},
    booktitle = {19th USENIX Conference on File and Storage Technologies (FAST 21)},
    year = {2021},
    isbn = {978-1-939133-20-5},
    pages = {203--216},
    url = {https://www.usenix.org/conference/fast21/presentation/mohan},
    publisher = {USENIX Association},
    month = feb
}
```