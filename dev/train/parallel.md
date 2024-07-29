# Parallel Training Strategies
*Here're some resources about parallel strategies in LLMs' multi-devices training*


### Integration of Parallelism

#### Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model

paper link: [here](https://arxiv.org/pdf/2201.11990.pdf)

citation:
```bibtex
@misc{smith2022using,
      title={Using DeepSpeed and Megatron to Train Megatron-Turing NLG 530B, A Large-Scale Generative Language Model}, 
      author={Shaden Smith and Mostofa Patwary and Brandon Norick and Patrick LeGresley and Samyam Rajbhandari and Jared Casper and Zhun Liu and Shrimai Prabhumoye and George Zerveas and Vijay Korthikanti and Elton Zhang and Rewon Child and Reza Yazdani Aminabadi and Julie Bernauer and Xia Song and Mohammad Shoeybi and Yuxiong He and Michael Houston and Saurabh Tiwary and Bryan Catanzaro},
      year={2022},
      eprint={2201.11990},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### Colossal-ai: A unified deep learning system for large-scale parallel training

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


#### GSPMD: General and Scalable Parallelization for ML Computation Graphs

paper link: [here](https://arxiv.org/pdf/2105.04663)

citation:

```bibtex
@misc{xu2021gspmd,
      title={GSPMD: General and Scalable Parallelization for ML Computation Graphs}, 
      author={Yuanzhong Xu and HyoukJoong Lee and Dehao Chen and Blake Hechtman and Yanping Huang and Rahul Joshi and Maxim Krikun and Dmitry Lepikhin and Andy Ly and Marcello Maggioni and Ruoming Pang and Noam Shazeer and Shibo Wang and Tao Wang and Yonghui Wu and Zhifeng Chen},
      year={2021},
      eprint={2105.04663},
      archivePrefix={arXiv},
      primaryClass={id='cs.DC' full_name='Distributed, Parallel, and Cluster Computing' is_active=True alt_name=None in_archive='cs' is_general=False description='Covers fault-tolerance, distributed algorithms, stabilility, parallel computation, and cluster computing. Roughly includes material in ACM Subject Classes C.1.2, C.1.4, C.2.4, D.1.3, D.4.5, D.4.7, E.1.'}
}
```

#### Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM (PTD-P)

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

#### DeepSpeed: System Optimizations Enable Training Deep Learning Models with Over 100 Billion Parameters

paper link: [here](https://dl.acm.org/doi/10.1145/3394486.3406703)

blog link: [here](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)

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


#### Gshard: Scaling giant models with conditional computation and automatic sharding

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



### Expert Parallelism (EP)


#### FlexMoE: Scaling Large-scale Sparse Pre-trained Model Training via Dynamic Device Placement

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

paper link: [here](https://dl.acm.org/doi/pdf/10.1145/3503221.3508418)

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


#### MegaBlocks: Efficient Sparse Training with Mixture-of-Experts (dMoE)

paper link: [here](https://arxiv.org/pdf/2211.15841.pdf)

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

paper link: [here](https://arxiv.org/pdf/2206.03382)

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


#### Deepspeed-moe: Advancing mixture-of-experts inference and training to power next-generation ai scale

paper link: [here](https://proceedings.mlr.press/v162/rajbhandari22a/rajbhandari22a.pdf)

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


#### Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity (SMoE)

paper link: [here](https://www.jmlr.org/papers/volume23/21-0998/21-0998.pdf)

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


### Context Parallelism (CP)


#### DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models

paper link: [here](https://arxiv.org/pdf/2309.14509)

blog link: [here](https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-ulysses/chinese/README.md)

github link: [here](https://github.com/microsoft/DeepSpeed/tree/master)

citation:

```bibtex
@misc{jacobs2023deepspeed,
      title={DeepSpeed Ulysses: System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models}, 
      author={Sam Ade Jacobs and Masahiro Tanaka and Chengming Zhang and Minjia Zhang and Shuaiwen Leon Song and Samyam Rajbhandari and Yuxiong He},
      year={2023},
      eprint={2309.14509},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### Sequence parallelism: Long sequence training from system perspective (Ring Self-Attention)

paper link: [here](https://arxiv.org/pdf/2105.13120)

citation:

```bibtex
@article{li2021sequence,
  title={Sequence parallelism: Long sequence training from system perspective},
  author={Li, Shenggui and Xue, Fuzhao and Baranwal, Chaitanya and Li, Yongbin and You, Yang},
  journal={arXiv preprint arXiv:2105.13120},
  year={2021}
}
```


### Pipeline Parallelism (PP)


#### Zero Bubble Pipeline Parallelism

paper link: [here](https://arxiv.org/pdf/2401.10241)

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

#### BPipe: Memory-Balanced Pipeline Parallelism for Training Large Language Models

paper link: [here](https://proceedings.mlr.press/v202/kim23l/kim23l.pdf)

citation:

```bibtex
@inproceedings{pmlr-v202-kim23l,
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

paper link: [here](https://arxiv.org/pdf/2107.06925)

citation:

```bibtex
@inproceedings{Li_2021, series={SC ’21},
   title={Chimera: efficiently training large-scale neural networks with bidirectional pipelines},
   url={http://dx.doi.org/10.1145/3458817.3476145},
   DOI={10.1145/3458817.3476145},
   booktitle={Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis},
   publisher={ACM},
   author={Li, Shigang and Hoefler, Torsten},
   year={2021},
   month=nov, collection={SC ’21} 
}
```


#### DAPPLE: A Pipelined Data Parallel Approach for Training Large Models

paper link: [here](https://arxiv.org/pdf/2007.01045)

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


#### Memory-Efficient Pipeline-Parallel DNN Training (PipeDream-2BW)

paper link: [here](https://arxiv.org/pdf/2006.09503)

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


#### GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism

paper link: [here](https://arxiv.org/pdf/1811.06965.pdf)

blog link: [here](https://www.deepspeed.ai/tutorials/pipeline/)

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

paper link: [here](https://arxiv.org/pdf/1806.03377)

citation:

```bibtex
@misc{harlap2018pipedream,
      title={PipeDream: Fast and Efficient Pipeline Parallel DNN Training}, 
      author={Aaron Harlap and Deepak Narayanan and Amar Phanishayee and Vivek Seshadri and Nikhil Devanur and Greg Ganger and Phil Gibbons},
      year={2018},
      eprint={1806.03377},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```


### Sequence Parallelism (SP)

#### Reducing Activation Recomputation in Large Transformer Models (SP)

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


### Tensor Parallelism (TP)


#### Maximizing Parallelism in Distributed Training for Huge Neural Networks (3D TP)

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

#### Tesseract: Parallelize the Tensor Parallelism Efficiently (2.5D TP)

paper link: [here](https://arxiv.org/pdf/2105.14500)

citation:

```bibtex
@inproceedings{Wang_2022, series={ICPP ’22},
   title={Tesseract: Parallelize the Tensor Parallelism Efficiently},
   url={http://dx.doi.org/10.1145/3545008.3545087},
   DOI={10.1145/3545008.3545087},
   booktitle={Proceedings of the 51st International Conference on Parallel Processing},
   publisher={ACM},
   author={Wang, Boxiang and Xu, Qifan and Bian, Zhengda and You, Yang},
   year={2022},
   month=aug, collection={ICPP ’22} }

```


#### An Efficient 2D Method for Training Super-Large Deep Learning Models (2D TP)

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


#### Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism (1D TP)

paper link: [here](https://arxiv.org/pdf/1909.08053.pdf)

github links:

|repo name| repo link|
|-|-|
|Megatron-LM|[here](https://github.com/NVIDIA/Megatron-LM)|
|Megatron-Deepspeed|[here](https://github.com/microsoft/Megatron-DeepSpeed)|


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


### Data Parallelism (DP)


#### ZeRO++: Extremely Efficient Collective Communication for Giant Model Training

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


#### Pytorch FSDP: experiences on scaling fully sharded data parallel

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


#### ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning

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


#### Automatic Cross-Replica Sharding of Weight Update in Data-Parallel Training

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


#### Distributed Data Parallelism (DDP)

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



