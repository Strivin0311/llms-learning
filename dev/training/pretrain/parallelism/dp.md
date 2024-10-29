

# Data Parallelism for LLMs Training
*Here're some resources about Data Parallelism for LLMs Training*
*Note that the modern data parallelism is beyond the traditional ddp and involves model sharding as well*


#### PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel

tag: `FSDP` | `VLDB23` | `Pytorch` | `Meta`

paper link: [here](https://dl.acm.org/doi/pdf/10.14778/3611540.3611569)

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


#### Automatic Cross-Replica Sharding of Weight Update in Data-Parallel Training

tag: `Automatic Cross-Replica Sharding` | `Google`

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



