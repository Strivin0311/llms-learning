# Mixed-Precision Training for LLMs
*Here're some resources about Mixed-Precision strategies, especially low-precision training for LLMs*
*Note that many of the methods here can be shared with the ones in quantization*


#### eXmY: A Data Type and Technique for Arbitrary Bit Precision Quantization

tag: `eXmY` | `Google`

paper link: [here](https://arxiv.org/pdf/2405.13938)

citation:

```bibtex
@misc{agrawal2024exmydatatypetechnique,
      title={eXmY: A Data Type and Technique for Arbitrary Bit Precision Quantization}, 
      author={Aditya Agrawal and Matthew Hedlund and Blake Hechtman},
      year={2024},
      eprint={2405.13938},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.13938}, 
}
```


#### A Stochastic Rounding-Enabled Low-Precision Floating-Point MAC for DNN Training

tag: `Optimized SR` | `Stochastic Rounding`

paper link: [here](https://arxiv.org/pdf/2404.14010)

citation:

```bibtex
@misc{ali2024stochasticroundingenabledlowprecisionfloatingpoint,
      title={A Stochastic Rounding-Enabled Low-Precision Floating-Point MAC for DNN Training}, 
      author={Sami Ben Ali and Silviu-Ioan Filip and Olivier Sentieys},
      year={2024},
      eprint={2404.14010},
      archivePrefix={arXiv},
      primaryClass={cs.AR},
      url={https://arxiv.org/abs/2404.14010}, 
}
```


#### FP8-LM: Training FP8 Large Language Models

tag: `FP8-LM` | `FP8 Optimizer` | `Microsoft`

paper link: [here](https://arxiv.org/pdf/2310.18313.pdf)

code link: [here](https://github.com/Azure/MS-AMP)

citation:

```bibtex
@misc{peng2023fp8lm,
      title={FP8-LM: Training FP8 Large Language Models}, 
      author={Houwen Peng and Kan Wu and Yixuan Wei and Guoshuai Zhao and Yuxiang Yang and Ze Liu and Yifan Xiong and Ziyue Yang and Bolin Ni and Jingcheng Hu and Ruihang Li and Miaosen Zhang and Chen Li and Jia Ning and Ruizhe Wang and Zheng Zhang and Shuguang Liu and Joe Chau and Han Hu and Peng Cheng},
      year={2023},
      eprint={2310.18313},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### Microscaling Data Formats for Deep Learning

tag: `Microscaling` | `MX` | `MXFP8` | `MXFP6` | `MXFP4` | `Microsoft`

paper link: [here](https://arxiv.org/pdf/2310.10537)

spec link: [here](https://www.opencompute.org/documents/ocp-microscaling-formats-mx-v1-0-spec-final-pdf)

citation:

```bibtex
@misc{rouhani2023microscalingdataformatsdeep,
      title={Microscaling Data Formats for Deep Learning}, 
      author={Bita Darvish Rouhani and Ritchie Zhao and Ankit More and Mathew Hall and Alireza Khodamoradi and Summer Deng and Dhruv Choudhary and Marius Cornea and Eric Dellinger and Kristof Denolf and Stosic Dusan and Venmugil Elango and Maximilian Golub and Alexander Heinecke and Phil James-Roxby and Dharmesh Jani and Gaurav Kolhe and Martin Langhammer and Ada Li and Levi Melnick and Maral Mesmakhosroshahi and Andres Rodriguez and Michael Schulte and Rasoul Shafipour and Lei Shao and Michael Siu and Pradeep Dubey and Paulius Micikevicius and Maxim Naumov and Colin Verrilli and Ralph Wittig and Doug Burger and Eric Chung},
      year={2023},
      eprint={2310.10537},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2310.10537}, 
}
```


#### Training and inference of large language models using 8-bit floating point

tag: `FP8 Scaling Bias` | `Graphcore Research`

paper link: [here](https://arxiv.org/pdf/2309.17224)

citation:

```bibtex
@misc{perez2023training,
      title={Training and inference of large language models using 8-bit floating point}, 
      author={Sergio P. Perez and Yan Zhang and James Briggs and Charlie Blake and Josh Levy-Kramer and Paul Balanca and Carlo Luschi and Stephen Barlow and Andrew William Fitzgibbon},
      year={2023},
      eprint={2309.17224},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### Stable and low-precision training for large-scale vision-language models

tag: `SwitchBlock` | `StableAdamW` | `NIPS23` | `Meta` | `Allen AI` | `LAION`

paper link: [here](https://proceedings.neurips.cc/paper_files/paper/2023/file/20bd42d82998bc61732c00452228e814-Supplemental-Conference.pdf)

citation:

```bibtex
@misc{wortsman2023stablelowprecisiontraininglargescale,
      title={Stable and low-precision training for large-scale vision-language models}, 
      author={Mitchell Wortsman and Tim Dettmers and Luke Zettlemoyer and Ari Morcos and Ali Farhadi and Ludwig Schmidt},
      year={2023},
      eprint={2304.13013},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2304.13013}, 
}
```


#### Unit Scaling: Out-of-the-Box Low-Precision Training

tag: `Unit Scaling` | `SNR Analysis` | `Graphcore Research`

paper link: [here](https://arxiv.org/pdf/2303.11257)

citation:

```bibtex
@misc{blake2023unitscalingoutoftheboxlowprecision,
      title={Unit Scaling: Out-of-the-Box Low-Precision Training}, 
      author={Charlie Blake and Douglas Orr and Carlo Luschi},
      year={2023},
      eprint={2303.11257},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2303.11257}, 
}
```


#### NVIDIA Transformer Engine: Accelerating PyTorch Training Workloads with FP8

tag: `TE FP8` | `Transformer-Engine` | `Nvidia`

blog link: [here](https://towardsdatascience.com/accelerating-pytorch-training-workloads-with-fp8-5a5123aec7d7)

docs link: [here](https://docs.nvidia.com/deeplearning/transformer-engine/index.html)

code link: [here](https://github.com/NVIDIA/TransformerEngine)

citation:

```bibtex
@misc{NVIDIA2023TransformerEngine,
  title={NVIDIA Transformer Engine: Accelerating PyTorch Training Workloads with FP8 (TE)},
  author={Chaim Rand, and NVIDIA},
  howpublished = {\url{https://github.com/NVIDIA/TransformerEngine}},
  year={2023},
}
```


#### NVIDIA Train With Mixed Precision

tag: `Mixed Precision` | `Nvidia`

docs link: [here](https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html)

citation:

```bibtex
@manual{nvidia2024mixed,
  title = {Train With Mixed Precision},
  author= {{NVIDIA Corporation}},
  month = {February},
  year  = {2023},
  howpublished = {\url{https://docs.nvidia.com/deeplearning/performance/mixed-precision-training/index.html}}
}
```


#### FP8 Quantization: The Power of the Exponent

tag: `FP8 Quantization` | `Qualcomm AI Research`

paper link: [here](https://arxiv.org/pdf/2208.09225.pdf)

code link: [here](https://github.com/Qualcomm-AI-research/FP8-quantization)

citation:

```bibtex
@misc{kuzmin2024fp8,
      title={FP8 Quantization: The Power of the Exponent}, 
      author={Andrey Kuzmin and Mart Van Baalen and Yuwei Ren and Markus Nagel and Jorn Peters and Tijmen Blankevoort},
      year={2024},
      eprint={2208.09225},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### 8-bit Optimizers via Block-wise Quantization

tag: `FP8 Optimizer` | `Blockwise Quantization` | `Dynamic Tree Quantization` | `Meta` | `University of Washington`

paper link: [here](https://arxiv.org/pdf/2110.02861)

citation:

```bibtex
@misc{dettmers20228bit,
      title={8-bit Optimizers via Block-wise Quantization}, 
      author={Tim Dettmers and Mike Lewis and Sam Shleifer and Luke Zettlemoyer},
      year={2022},
      eprint={2110.02861},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### Hybrid 8-bit Floating Point (HFP8) Training and Inference for Deep Neural Networks

tag: `HFP8` | `Hybrid FP8` | `NIPS19` | `IBM Research`      

paper link: [here](https://proceedings.neurips.cc/paper_files/paper/2019/file/65fc9fb4897a89789352e211ca2d398f-Paper.pdf)

citation:

```bibtex
@inproceedings{NEURIPS2019_65fc9fb4,
      author = {Sun, Xiao and Choi, Jungwook and Chen, Chia-Yu and Wang, Naigang and Venkataramani, Swagath and Srinivasan, Vijayalakshmi (Viji) and Cui, Xiaodong and Zhang, Wei and Gopalakrishnan, Kailash},
      booktitle = {Advances in Neural Information Processing Systems},
      editor = {H. Wallach and H. Larochelle and A. Beygelzimer and F. d\textquotesingle Alch\'{e}-Buc and E. Fox and R. Garnett},
      pages = {},
      publisher = {Curran Associates, Inc.},
      title = {Hybrid 8-bit Floating Point (HFP8) Training and Inference for Deep Neural Networks},
      url = {https://proceedings.neurips.cc/paper_files/paper/2019/file/65fc9fb4897a89789352e211ca2d398f-Paper.pdf},
      volume = {32},
      year = {2019}
}  
```

#### BFloat16: The secret to high performance on Cloud TPUs

tag: `BF16` | `Google`

blog link: [here](https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus)

citation:

```bibtex
@misc{shibo2019bfloat16,
  author = {Shibo Wang and Pankaj Kanwar},
  title  = {bfloat16: The secret to high performance on Cloud TPUs},
  year   = {2019},
  howpublished = {\url{https://cloud.google.com/blog/products/ai-machine-learning/bfloat16-the-secret-to-high-performance-on-cloud-tpus}}
}
```


#### Training Deep Neural Networks with 8-bit Floating Point Numbers

tag: `FP8-E5M2` | `Chunk-based Accumulation` | `Stochastic Rounding` | `IBM Research`

paper link: [here](https://arxiv.org/pdf/1812.08011)

citation:

```bibtex
@misc{wang2018training,
      title={Training Deep Neural Networks with 8-bit Floating Point Numbers}, 
      author={Naigang Wang and Jungwook Choi and Daniel Brand and Chia-Yu Chen and Kailash Gopalakrishnan},
      year={2018},
      eprint={1812.08011},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### Mixed Precision Training

tag: `FP16` | `Loss Scaling` | `ICLR18` | `Baidu` | `Nvidia`

paper link: [here](https://arxiv.org/pdf/1710.03740)

citation:

```bibtex
@misc{micikevicius2018mixed,
      title={Mixed Precision Training}, 
      author={Paulius Micikevicius and Sharan Narang and Jonah Alben and Gregory Diamos and Erich Elsen and David Garcia and Boris Ginsburg and Michael Houston and Oleksii Kuchaiev and Ganesh Venkatesh and Hao Wu},
      year={2018},
      eprint={1710.03740},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```