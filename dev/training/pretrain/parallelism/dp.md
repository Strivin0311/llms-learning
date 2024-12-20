# Data Parallelism for LLMs Training
*Here're some resources about Data Parallelism for LLMs Training*
*Note that the modern data parallelism is beyond the traditional ddp and involves model sharding as well*


#### LuWu: An End-to-End In-Network Out-of-Core Optimizer for 100B-Scale Model-in-Network Data-Parallel Training on Distributed GPUs

tag: `Luwu` | `DP` | `Data Parallelism` | `Zhejiang University`

paper link: [here](https://arxiv.org/pdf/2409.00918)

citation:

```bibtex
@misc{sun2024luwuendtoendinnetworkoutofcore,
      title={LuWu: An End-to-End In-Network Out-of-Core Optimizer for 100B-Scale Model-in-Network Data-Parallel Training on Distributed GPUs}, 
      author={Mo Sun and Zihan Yang and Changyue Liao and Yingtao Li and Fei Wu and Zeke Wang},
      year={2024},
      eprint={2409.00918},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2409.00918}, 
}
```

#### PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel

tag: `FSDP` | `FSDP-v1` | `FSDP-v2` | `VLDB23` | `Pytorch` | `Meta`

paper link: [here](https://arxiv.org/pdf/2304.11277)

blog link: [here](https://pytorch.org/blog/introducing-pytorch-fully-sharded-data-parallel-api/)

docs link: [here](https://pytorch.org/docs/stable/fsdp.html)

notes link: [here](https://pytorch.org/docs/stable/notes/fsdp.html)

follow-up work: [here](https://github.com/pytorch/torchtitan/blob/main/docs/fsdp.md)

citation:

```bibtex
@article{zhao2023pytorch,
  title={Pytorch FSDP: experiences on scaling fully sharded data parallel},
  author={Zhao, Yanli and Gu, Andrew and Varma, Rohan and Luo, Liang and Huang, Chien-Chin and Xu, Min and Wright, Less and Shojanazeri, Hamid and Ott, Myle and Shleifer, Sam and others},
  journal={arXiv preprint arXiv:2304.11277},
  year={2023}
}
```


#### MiCS: near-linear scaling for training gigantic model on public cloud

tag: `MiCS` | `VLDB22` | `Amazon` | `JHU`

paper link: [here](https://www.vldb.org/pvldb/vol16/p37-zhang.pdf)

citation:

```bibtex
@misc{zhang2022micsnearlinearscalingtraining,
      title={MiCS: Near-linear Scaling for Training Gigantic Model on Public Cloud}, 
      author={Zhen Zhang and Shuai Zheng and Yida Wang and Justin Chiu and George Karypis and Trishul Chilimbi and Mu Li and Xin Jin},
      year={2022},
      eprint={2205.00119},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2205.00119}, 
}
```


#### Automatic Cross-Replica Sharding of Weight Update in Data-Parallel Training

tag: `Cross-Replica` | `XLA` | `Google`

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


#### ZeRO: Memory Optimizations Toward Training Trillion Parameter Models

tag: `ZERO` | `ZERO-1` | `ZERO-2` | `ZERO-3` | `DeepSpeed` | `SC20` | `Microsoft`

paper link: [here](https://dl.acm.org/doi/pdf/10.5555/3433701.3433727)

blog link: [here](https://www.microsoft.com/en-us/research/blog/zero-deepspeed-new-system-optimizations-enable-training-models-with-over-100-billion-parameters/)

citation:

```bibtex
@inproceedings{rajbhandari2020zero,
      author = {Rajbhandari, Samyam and Rasley, Jeff and Ruwase, Olatunji and He, Yuxiong},
      title = {ZeRO: memory optimizations toward training trillion parameter models},
      year = {2020},
      isbn = {9781728199986},
      publisher = {IEEE Press},
      articleno = {20},
      numpages = {16},
      location = {Atlanta, Georgia},
      series = {SC '20}
}
```


#### Pytorch Distributed Data Parallelism

tag: `DDP` | `Pytorch` | `Meta`

blog link: [here](https://pytorch.org/docs/master/notes/ddp.html)

docs link: [here](https://pytorch.org/docs/master/generated/torch.nn.parallel.DistributedDataParallel.html)

tutorial link: [here](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

citation:

```bibtex
@misc{pytorch2019ddp,
      title={Distributed Data Parallel},
      author={PyTorch contributors},
      year={2019},
      url={https://pytorch.org/docs/master/notes/ddp.html}
}
```