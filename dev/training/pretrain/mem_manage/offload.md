# Offloading Strategies for LLMs Training
*Here're some resources about Offloading Strategies for LLMs Training*


#### Accelerating the Training of Large Language Models using Efficient Activation Rematerialization and Optimal Hybrid Parallelism

tag: `Pipeline-Parallel-Aware Offloading` | `ATC24` | `Kuaishou Technology`

paper link: [here](https://www.usenix.org/system/files/atc24-yuan.pdf)

slides link: [here](https://www.usenix.org/system/files/atc24_slides-yuan.pdf)

citation:

```bibtex
@inproceedings {yuan2024pipelineparallelawareoffloading,
      author = {Tailing Yuan and Yuliang Liu and Xucheng Ye and Shenglong Zhang and Jianchao Tan and Bin Chen and Chengru Song and Di Zhang},
      title = {Accelerating the Training of Large Language Models using Efficient Activation Rematerialization and Optimal Hybrid Parallelism},
      booktitle = {2024 USENIX Annual Technical Conference (USENIX ATC 24)},
      year = {2024},
      isbn = {978-1-939133-41-0},
      address = {Santa Clara, CA},
      pages = {545--561},
      url = {https://www.usenix.org/conference/atc24/presentation/yuan},
      publisher = {USENIX Association},
      month = jul
}
```

#### Efficient and Economic Large Language Model Inference with Attention Offloading

tag: `Attention Offloading` | `Tsinghua University`

paper link: [here](https://arxiv.org/pdf/2405.01814v1)

citation:

```bibtex
@misc{chen2024efficienteconomiclargelanguage,
      title={Efficient and Economic Large Language Model Inference with Attention Offloading}, 
      author={Shaoyuan Chen and Yutong Lin and Mingxing Zhang and Yongwei Wu},
      year={2024},
      eprint={2405.01814},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.01814}, 
}
```


#### Adding NVMe SSDs to Enable and Accelerate 100B Model Fine-tuning on a Single GPU

tag: `Lohan` | `ICDE25` | `Zhejiang University` | `HKU`

paper link: [here](https://arxiv.org/pdf/2403.06504)

github link: [here](https://github.com/Crispig/Ratel)

citation:

```bibtex
@misc{liao2024addingnvmessdsenable,
      title={Adding NVMe SSDs to Enable and Accelerate 100B Model Fine-tuning on a Single GPU}, 
      author={Changyue Liao and Mo Sun and Zihan Yang and Kaiqi Chen and Binhang Yuan and Fei Wu and Zeke Wang},
      year={2024},
      eprint={2403.06504},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2403.06504}, 
}
```

#### NVIDIA Transformer-Engine CPU Offloading

tag: `TE-Offloading` | `Transformer-Engine` | `NVIDIA`

github link: [here](https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/cpu_offload.py)

docs link: [here](https://docs.nvidia.com/deeplearning/transformer-engine/user-guide/api/pytorch.html?highlight=offload#transformer_engine.pytorch.get_cpu_offload_context)

citation: 

```bibtex
@misc{transformerenginecpuoffload2024nvidia,
  author = {NVIDIA},
  title  = {NVIDIA Transformer-Engine CPU Offloading},
  year   = {2024},
  howpublished = {\url{https://github.com/NVIDIA/TransformerEngine/blob/main/transformer_engine/pytorch/cpu_offload.py}},
}
```


#### STR: Hybrid Tensor Re-Generation to Break Memory Wall for DNN Training

tag: `STR` | `TPDS23` | `Tsinghua University`

paper link: [here](https://ieeexplore.ieee.org/document/10098636)

citation:

```bibtex
@article{zong2023str,
  author={Zong, Zan and Lin, Li and Lin, Leilei and Wen, Lijie and Sun, Yu},
  journal={IEEE Transactions on Parallel and Distributed Systems}, 
  title={STR: Hybrid Tensor Re-Generation to Break Memory Wall for DNN Training}, 
  year={2023},
  volume={34},
  number={8},
  pages={2403-2418},
  keywords={Tensors;Graphics processing units;Training;Optimization;Costs;Bandwidth;Memory management;DNN training;offload memory;recomputation;rematerialization;swap},
  doi={10.1109/TPDS.2023.3266110}
}
```

#### MPress: Democratizing Billion-Scale Model Training on Multi-GPU Servers via Memory-Saving Inter-Operator Parallelism

tag: `MPress` | `HPCA23` | `USTC`

paper link: [here](https://ieeexplore.ieee.org/document/10071077)

citation:

```bibtex
@inproceedings{zhou2023mpress,
  author={Zhou, Quan and Wang, Haiquan and Yu, Xiaoyan and Li, Cheng and Bai, Youhui and Yan, Feng and Xu, Yinlong},
  booktitle={2023 IEEE International Symposium on High-Performance Computer Architecture (HPCA)}, 
  title={MPress: Democratizing Billion-Scale Model Training on Multi-GPU Servers via Memory-Saving Inter-Operator Parallelism}, 
  year={2023},
  volume={},
  number={},
  pages={556-569},
  keywords={Training;Performance evaluation;Tensors;Costs;Computational modeling;Graphics processing units;Parallel processing;Inter-Operator Parallelism;DNN Training;Swap;Recomputation},
  doi={10.1109/HPCA56546.2023.10071077}
}
```

#### STRONGHOLD: Fast and Affordable Billion-Scale Deep Learning Model Training

tag: `STRONGHOLD` | `SC22` | `Alibaba Group`

paper link: [here](https://ieeexplore.ieee.org/document/10046110)

citation:

```bibtex
@inproceedings{sun2022stronghold,
  author={Sun, Xiaoyang and Wang, Wei and Qiu, Shenghao and Yang, Renyu and Huang, Songfang and Xu, Jie and Wang, Zheng},
  booktitle={SC22: International Conference for High Performance Computing, Networking, Storage and Analysis}, 
  title={STRONGHOLD: Fast and Affordable Billion-Scale Deep Learning Model Training}, 
  year={2022},
  volume={},
  number={},
  pages={1-17},
  keywords={Training;Deep learning;Codes;Computational modeling;Memory management;Graphics processing units;Random access memory;Deep learning;Distributed training;DNNs training acceleration},
  doi={10.1109/SC41404.2022.00076}
}
```


#### DELTA: Memory-Eficient Training via Dynamic Fine-Grained Recomputation and Swapping

tag: `DELTA` | `Tensor Swapping` | `Tensor Recomputation` | `ACM TACO24`

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


#### ZeRO-Infinity: Breaking the GPU Memory Wall for Extreme Scale Deep Learning

tag: `ZeRO-Infinity` | `NVMe SSD` | `Infinity Offload Engine` | `Microsoft`

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


#### ZeRO-Offload: Democratizing Billion-Scale Model Training

tag: `ZeRO-Offload` | `Microsoft`

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

#### SwapAdvisor: Pushing Deep Learning Beyond the GPU Memory Limit via Smart Swapping

tag: `SwapAdvisor` | `ASPLOS20` | `New York University`

paper link: [here](https://dl.acm.org/doi/pdf/10.1145/3373376.3378530)

citation:

```bibtex
@inproceedings{swapadvisor2020huang,
      author = {Huang, Chien-Chin and Jin, Gu and Li, Jinyang},
      title = {SwapAdvisor: Pushing Deep Learning Beyond the GPU Memory Limit via Smart Swapping},
      year = {2020},
      isbn = {9781450371025},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3373376.3378530},
      doi = {10.1145/3373376.3378530},
      booktitle = {Proceedings of the Twenty-Fifth International Conference on Architectural Support for Programming Languages and Operating Systems},
      pages = {1341–1355},
      numpages = {15},
      keywords = {scheduling and resource management, gpu, deep learning systems},
      location = {Lausanne, Switzerland},
      series = {ASPLOS '20}
}
```


#### Capuchin: Tensor-based GPU Memory Management for Deep Learning

tag: `Capuchin` | `ASPLOS20` | `MSRA`

paper link: [here](https://alchem.usc.edu/portal/static/download/capuchin.pdf)

citation:

```bibtex
@inproceedings{peng2020capuchin,
      author = {Peng, Xuan and Shi, Xuanhua and Dai, Hulin and Jin, Hai and Ma, Weiliang and Xiong, Qian and Yang, Fan and Qian, Xuehai},
      title = {Capuchin: Tensor-based GPU Memory Management for Deep Learning},
      year = {2020},
      isbn = {9781450371025},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3373376.3378505},
      doi = {10.1145/3373376.3378505},
      pages = {891–905},
      numpages = {15},
      keywords = {deep learning training, gpu memory management, tensor access},
      location = {Lausanne, Switzerland},
      series = {ASPLOS '20}
}
```


#### Training Large Neural Networks with Constant Memory using a New Execution Algorithm

tag: `L2L` | `EPS` | `Microsoft`

paper link: [here](https://arxiv.org/pdf/2002.05645)

citation:

```bibtex
@misc{pudipeddi2020training,
      title={Training Large Neural Networks with Constant Memory using a New Execution Algorithm}, 
      author={Bharadwaj Pudipeddi and Maral Mesmakhosroshahi and Jinwen Xi and Sujeeth Bharadwaj},
      year={2020},
      eprint={2002.05645},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

#### SuperNeurons: Dynamic GPU Memory Management for Training Deep Neural Networks

tag: `SuperNeurons` | `PPoPP18` | `MIT`

paper link: [here](https://dl.acm.org/doi/pdf/10.1145/3200691.3178491)

citation:

```bibtex
@inproceedings{li2018supernurons,
      author = {Wang, Linnan and Ye, Jinmian and Zhao, Yiyang and Wu, Wei and Li, Ang and Song, Shuaiwen Leon and Xu, Zenglin and Kraska, Tim},
      title = {Superneurons: dynamic GPU memory management for training deep neural networks},
      year = {2018},
      isbn = {9781450349826},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3178487.3178491},
      doi = {10.1145/3178487.3178491},
      pages = {41–53},
      numpages = {13},
      keywords = {runtime scheduling, neural networks, GPU memory management},
      location = {Vienna, Austria},
      series = {PPoPP '18}
}
```


#### vDNN: Virtualized Deep Neural Networks for Scalable, Memory-Efficient Neural Network Design

tag: `vDNN` | `Nvidia`

paper link: [here](https://arxiv.org/pdf/1602.08124)

citation:

```bibtex
@misc{rhu2016vdnn,
      title={vDNN: Virtualized Deep Neural Networks for Scalable, Memory-Efficient Neural Network Design}, 
      author={Minsoo Rhu and Natalia Gimelshein and Jason Clemons and Arslan Zulfiqar and Stephen W. Keckler},
      year={2016},
      eprint={1602.08124},
      archivePrefix={arXiv},
      primaryClass={cs.DC}
}
```
