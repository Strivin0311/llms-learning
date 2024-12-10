# Tensor Recomputation / Activation Checkpointing for LLMs Training
*Here're some resources about Tensor Recomputation / Activation Checkpointing for LLMs Training, to trade off computation for memory*



#### How Activation Checkpointing enables scaling up training deep learning models

tag: `Activation Checkpointing` | `PyTorch` | `Meta`

blog link: [here](https://medium.com/pytorch/how-activation-checkpointing-enables-scaling-up-training-deep-learning-models-7a93ae01ff2d)

docs link: [here](https://pytorch.org/docs/stable/checkpoint.html)

citation:

```bibtex
@article{beer2023pytorchcheckpointing,
  title = {How Activation Checkpointing Enables Scaling Up Training Deep Learning Models},
  author = {Yiftach Beer and Omri Bar},
  month = nov,
  year = {2023},
  url = {https://medium.com/pytorch/how-activation-checkpointing-enables-scaling-up-training-deep-learning-models-7a93ae01ff2d},
}
```


#### DELTA: Memory-Eficient Training via Dynamic Fine-Grained Recomputation and Swapping

tag: `DELTA` | `Tensor Swapping` | `Tensor Recomputation` | `ACM TACO24` | `NUDT` | `Xiamen University`

paper link: [here](https://dl.acm.org/doi/pdf/10.1145/3689338)

citation:

```bibtex
@misc{tang2022deltadynamicallyoptimizinggpu,
      title={DELTA: Dynamically Optimizing GPU Memory beyond Tensor Recomputation}, 
      author={Yu Tang and Chenyu Wang and Yufan Zhang and Yuliang Liu and Xingcheng Zhang and Linbo Qiao and Zhiquan Lai and Dongsheng Li},
      year={2022},
      eprint={2203.15980},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2203.15980}, 
}
```


#### Dynamic Tensor Rematerialization

tag: `DTR` | `Dynamic Tensor Rematerialization` | `ICLR21` | `University of Washington`

paper link: [here](https://arxiv.org/pdf/2006.09616)

github link: [here](https://github.com/uwsampl/dtr-prototype)

citation:

```bibtex
@misc{kirisame2021dynamictensorrematerialization,
      title={Dynamic Tensor Rematerialization}, 
      author={Marisa Kirisame and Steven Lyubomirsky and Altan Haan and Jennifer Brennan and Mike He and Jared Roesch and Tianqi Chen and Zachary Tatlock},
      year={2021},
      eprint={2006.09616},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2006.09616}, 
}
```


#### Training Deep Nets with Sublinear Memory Cost

tag: `SLiM` | `Sub-Linear Memory` | `MIT`

paper link: [here](https://arxiv.org/pdf/1604.06174)

citation:

```bibtex
@misc{chen2016trainingdeepnetssublinear,
      title={Training Deep Nets with Sublinear Memory Cost}, 
      author={Tianqi Chen and Bing Xu and Chiyuan Zhang and Carlos Guestrin},
      year={2016},
      eprint={1604.06174},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1604.06174}, 
}
```