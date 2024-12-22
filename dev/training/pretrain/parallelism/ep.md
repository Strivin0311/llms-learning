# Expert Parallelism for Sparse-MoE LLMs Training
*Here're some resources about Expert Parallelism for Sparse-MoE LLMs Training*
*Note that some of the methods below might be general techniques for MoE training beyond expert parallelism, and even overlapped with MoE modeling*


#### MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs

tag: `MoE-Lightning` | `UC Berkeley`

paper link: [here](https://arxiv.org/pdf/2411.11217)

citation:

```bibtex
@misc{cao2024moelightninghighthroughputmoeinference,
      title={MoE-Lightning: High-Throughput MoE Inference on Memory-constrained GPUs}, 
      author={Shiyi Cao and Shu Liu and Tyler Griggs and Peter Schafhalter and Xiaoxuan Liu and Ying Sheng and Joseph E. Gonzalez and Matei Zaharia and Ion Stoica},
      year={2024},
      eprint={2411.11217},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2411.11217}, 
}
```


#### LSH-MoE: Communication-efficient MoE Training via Locality-Sensitive Hashing

tag: `LSH-MoE` | `LSH` | `NIPS24` | `ByteDance` | `Peking University`

paper link: [here](https://arxiv.org/pdf/2411.08446)

citation:

```bibtex
@misc{nie2024lshmoecommunicationefficientmoetraining,
      title={LSH-MoE: Communication-efficient MoE Training via Locality-Sensitive Hashing}, 
      author={Xiaonan Nie and Qibin Liu and Fangcheng Fu and Shenhan Zhu and Xupeng Miao and Xiaoyang Li and Yang Zhang and Shouda Liu and Bin Cui},
      year={2024},
      eprint={2411.08446},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2411.08446}, 
}
```


#### Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models

tag: `Skywork MoE` | `Skywork Team, Kunlun Inc`

paper link: [here](https://arxiv.org/pdf/2406.06563)

github link: [here](https://github.com/SkyworkAI/Skywork-MOE)

model links:

|model name|link|
|-|-|
|Skywork-MoE-Base-FP8|[here](https://huggingface.co/Skywork/Skywork-MoE-Base-FP8)|
|Skywork-MoE-Base|[here](https://huggingface.co/Skywork/Skywork-MoE-Base)|

citation:

```bibtex
@misc{wei2024skyworkmoedeepdivetraining,
      title={Skywork-MoE: A Deep Dive into Training Techniques for Mixture-of-Experts Language Models}, 
      author={Tianwen Wei and Bo Zhu and Liang Zhao and Cheng Cheng and Biye Li and Weiwei Lü and Peng Cheng and Jianhao Zhang and Xiaoyu Zhang and Liang Zeng and Xiaokun Wang and Yutuan Ma and Rui Hu and Shuicheng Yan and Han Fang and Yahui Zhou},
      year={2024},
      eprint={2406.06563},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.06563}, 
}
```

#### FlexMoE: Scaling Large-scale Sparse Pre-trained Model Training via Dynamic Device Placement

tag: `FlexMoE` | `Dynamic Device Placement` | `ACM MOD23` | `Peking University` | `CMU` | `Microsoft`

paper link: [here](https://arxiv.org/pdf/2304.03946)

citation:

```bibtex
@article{nie2023flexmoe,
  title={FlexMoE: Scaling Large-scale Sparse Pre-trained Model Training via Dynamic Device Placement},
  author={Nie, Xiaonan and Miao, Xupeng and Wang, Zilong and Yang, Zichao and Xue, Jilong and Ma, Lingxiao and Cao, Gang and Cui, Bin},
  journal={Proceedings of the ACM on Management of Data},
  volume={1},
  number={1},
  pages={1--19},
  year={2023},
  publisher={ACM New York, NY, USA}
}
```

#### Accelerating Distributed MoE Training and Inference with Lina

tag: `Lina` | `ATC23` | `ByteDance`

paper link: [here](https://www.usenix.org/system/files/atc23-li-jiamin.pdf)

citation:

```bibtex
@inproceedings{li2023accelerating,
  title={Accelerating distributed $\{$MoE$\}$ training and inference with lina},
  author={Li, Jiamin and Jiang, Yimin and Zhu, Yibo and Wang, Cong and Xu, Hong},
  booktitle={2023 USENIX Annual Technical Conference (USENIX ATC 23)},
  pages={945--959},
  year={2023}
}
```

#### FasterMoE: modeling and optimizing training of large-scale dynamic pre-trained models

tag: `FasterMoE` | `PPoPP22` | `Tsinghua University`

paper link: [here](https://dl.acm.org/doi/pdf/10.1145/3503221.3508418)

github link: [here](https://github.com/thu-pacman/FasterMoE)

citation:

```bibtex
@inproceedings{he2022fastermoe,
  title={FasterMoE: modeling and optimizing training of large-scale dynamic pre-trained models},
  author={He, Jiaao and Zhai, Jidong and Antunes, Tiago and Wang, Haojie and Luo, Fuwen and Shi, Shangfeng and Li, Qin},
  booktitle={Proceedings of the 27th ACM SIGPLAN Symposium on Principles and Practice of Parallel Programming},
  pages={120--134},
  year={2022}
}
```


#### MegaBlocks: Efficient Sparse Training with Mixture-of-Experts

tag: `MegaBlocks` | `dMoE` | `MLSys23` | `Stanford University` | `Google` | `Microsoft`

paper link: [here](https://proceedings.mlsys.org/paper_files/paper/2023/file/5a54f79333768effe7e8927bcccffe40-Paper-mlsys2023.pdf)

github link: [here](https://github.com/stanford-futuredata/megablocks)

citation:

```bibtex
@misc{gale2022megablocks,
      title={MegaBlocks: Efficient Sparse Training with Mixture-of-Experts}, 
      author={Trevor Gale and Deepak Narayanan and Cliff Young and Matei Zaharia},
      year={2022},
      eprint={2211.15841},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### Tutel: Adaptive Mixture-of-Experts at Scale

tag: `Tutel` | `MLSys23` | `Microsoft`

paper link: [here](https://arxiv.org/pdf/2206.03382)

slides link: [here](https://mlsys.org/media/mlsys-2023/Slides/2477.pdf)

github link: [here](https://github.com/microsoft/tutel)

citation:

```bibtex
@misc{hwang2023tutel,
      title={Tutel: Adaptive Mixture-of-Experts at Scale}, 
      author={Changho Hwang and Wei Cui and Yifan Xiong and Ziyue Yang and Ze Liu and Han Hu and Zilong Wang and Rafael Salas and Jithin Jose and Prabhat Ram and Joe Chau and Peng Cheng and Fan Yang and Mao Yang and Yongqiang Xiong},
      year={2023},
      eprint={2206.03382},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```


#### DeepSpeed-MoE: Advancing mixture-of-experts inference and training to power next-generation ai scale

tag: `DeepSpeed-MoE` | `ICML23` | `Microsoft`

paper link: [here](https://proceedings.mlr.press/v162/rajbhandari22a/rajbhandari22a.pdf)

github link: [here](https://github.com/microsoft/DeepSpeed)

citation:

```bibtex
@inproceedings{rajbhandari2022deepspeed,
  title={Deepspeed-moe: Advancing mixture-of-experts inference and training to power next-generation ai scale},
  author={Rajbhandari, Samyam and Li, Conglong and Yao, Zhewei and Zhang, Minjia and Aminabadi, Reza Yazdani and Awan, Ammar Ahmad and Rasley, Jeff and He, Yuxiong},
  booktitle={International Conference on Machine Learning},
  pages={18332--18346},
  year={2022},
  organization={PMLR}
}
```


#### Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity

tag: `Switch Transformer` | `JMLR23` | `Google`

paper link: [here](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf)

github link: [here](https://github.com/tensorflow/mesh/blob/master/mesh_tensorflow/transformer/moe.py)

citation:

```bibtex
@article{fedus2022switch,
  title={Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity},
  author={Fedus, William and Zoph, Barret and Shazeer, Noam},
  journal={The Journal of Machine Learning Research},
  volume={23},
  number={1},
  pages={5232--5270},
  year={2022},
  publisher={JMLRORG}
}
```

#### GShard: Scaling Giant Models with Conditional Computation and Automatic Sharding

tag: `GShard` | `ICLR21` | `Google`

paper link: [here](https://arxiv.org/pdf/2006.16668.pdf)

citation:

```bibtex
@article{lepikhin2020gshard,
  title={Gshard: Scaling giant models with conditional computation and automatic sharding},
  author={Lepikhin, Dmitry and Lee, HyoukJoong and Xu, Yuanzhong and Chen, Dehao and Firat, Orhan and Huang, Yanping and Krikun, Maxim and Shazeer, Noam and Chen, Zhifeng},
  journal={arXiv preprint arXiv:2006.16668},
  year={2020}
}
```


