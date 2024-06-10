# Mixture-of-Experts (MoE)
*Here're some resources about Mixture-of-Experts (MoE) structure design of LLMs*


#### Fast Inference of Mixture-of-Experts Language Models with Offloading [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2312.17238.pdf)

github link: [here](https://github.com/dvmazur/mixtral-offloading)

citation: 
```bibtex
@misc{eliseev2023fast,
      title={Fast Inference of Mixture-of-Experts Language Models with Offloading}, 
      author={Artyom Eliseev and Denis Mazur},
      year={2023},
      eprint={2312.17238},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

#### SwitchHead: Accelerating Transformers with Mixture-of-Experts Attention [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2312.07987v2.pdf)

citation:

```bibtex
@article{csordas2023switchhead,
  title={SwitchHead: Accelerating Transformers with Mixture-of-Experts Attention},
  author={Csord{\'a}s, R{\'o}bert and Pi{\k{e}}kos, Piotr and Irie, Kazuki},
  journal={arXiv preprint arXiv:2312.07987},
  year={2023}
}
```


#### Mixtral of experts: A high quality Sparse Mixture-of-Experts [`READ`]

blog link: [here](https://mistral.ai/news/mixtral-of-experts/)

model links: 

|model name|link|
|-|-|
|Mixtral-SlimOrca-8x7B|[here](https://huggingface.co/Open-Orca/Mixtral-SlimOrca-8x7B)|
|Mixtral-8x7B-Instruct-v0.1|[here](https://huggingface.co/mistralai/Mixtral-8x7B-Instruct-v0.1)|
|Mixtral-8x7B-v0.1|[here](https://huggingface.co/mistralai/Mixtral-8x7B-v0.1)|

citation:
```bibtex
@misc{mixtral_model,
  author = {Mistral AI},
  title = {Mixtral of Experts: A High-Quality Sparse Mixture-of-Experts},
  year = {2023},
  url = {\url{https://mistral.ai/news/mixtral-of-experts/}}
}
```


#### Memory Augmented Language Models through Mixture of Word Experts [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2311.10768)

citation:

```bibtex
@article{santos2023memory,
  title={Memory Augmented Language Models through Mixture of Word Experts},
  author={Santos, Cicero Nogueira dos and Lee-Thorp, James and Noble, Isaac and Chang, Chung-Ching and Uthus, David},
  journal={arXiv preprint arXiv:2311.10768},
  year={2023}
}
```


#### QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2310.16795.pdf)

citation:
```bibtex
@misc{frantar2023qmoe,
      title={QMoE: Practical Sub-1-Bit Compression of Trillion-Parameter Models}, 
      author={Elias Frantar and Dan Alistarh},
      year={2023},
      eprint={2310.16795},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### From sparse to soft mixtures of experts (Soft MoE) [`READ`]

paper link: [here](https://arxiv.org/pdf/2308.00951)

citation:

```bibtex
@article{puigcerver2023sparse,
  title={From sparse to soft mixtures of experts},
  author={Puigcerver, Joan and Riquelme, Carlos and Mustafa, Basil and Houlsby, Neil},
  journal={arXiv preprint arXiv:2308.00951},
  year={2023}
}
```


#### OpenMoE: A family of open-sourced Mixture-of-Experts (MoE) Large Language Models [`UNREAD`]

blog link: [here](https://xuefuzhao.notion.site/Aug-2023-OpenMoE-v0-2-Release-43808efc0f5845caa788f2db52021879)

github link: [here](https://github.com/XueFuzhao/OpenMoE)

citation:
```bibtex
@misc{openmoe2023,
  author = {Fuzhao Xue, Zian Zheng, Yao Fu, Jinjie Ni, Zangwei Zheng, Wangchunshu Zhou and Yang You},
  title = {OpenMoE: Open Mixture-of-Experts Language Models},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/XueFuzhao/OpenMoE}},
}
```


#### Accelerating distributed {MoE} training and inference with lina [`UNREAD`]

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


#### AdaMV-MoE: Adaptive Multi-Task Vision Mixture-of-Experts [`UNREAD`]

paper link: [here](http://openaccess.thecvf.com/content/ICCV2023/papers/Chen_AdaMV-MoE_Adaptive_Multi-Task_Vision_Mixture-of-Experts_ICCV_2023_paper.pdf)

citation:
```bibtex
@inproceedings{chen2023adamv,
  title={AdaMV-MoE: Adaptive Multi-Task Vision Mixture-of-Experts},
  author={Chen, Tianlong and Chen, Xuxi and Du, Xianzhi and Rashwan, Abdullah and Yang, Fan and Chen, Huizhong and Wang, Zhangyang and Li, Yeqing},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={17346--17357},
  year={2023}
}
```


#### FlexMoE: Scaling Large-scale Sparse Pre-trained Model Training via Dynamic Device Placement [`UNREAD`]

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



#### MegaBlocks: Efficient Sparse Training with Mixture-of-Experts (dMoE) [`READ`]

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

#### Tutel: Adaptive Mixture-of-Experts at Scale [`UNREAD`]

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

#### Evomoe: An evolutional mixture-of-experts training framework via dense-to-sparse gate [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2112.14397)

citation:

```bibtex
@article{nie2021evomoe,
  title={Evomoe: An evolutional mixture-of-experts training framework via dense-to-sparse gate},
  author={Nie, Xiaonan and Miao, Xupeng and Cao, Shijie and Ma, Lingxiao and Liu, Qibin and Xue, Jilong and Miao, Youshan and Liu, Yi and Yang, Zhi and Cui, Bin},
  journal={arXiv preprint arXiv:2112.14397},
  year={2021}
}
```


#### Mixture of Attention Heads: Selecting Attention Heads Per Token (MoA) [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2210.05144)

citation:

```bibtex
@misc{zhang2022mixture,
      title={Mixture of Attention Heads: Selecting Attention Heads Per Token}, 
      author={Xiaofeng Zhang and Yikang Shen and Zeyu Huang and Jie Zhou and Wenge Rong and Zhang Xiong},
      year={2022},
      eprint={2210.05144},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

#### FasterMoE: modeling and optimizing training of large-scale dynamic pre-trained models [`UNREAD`]

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

#### Deepspeed-moe: Advancing mixture-of-experts inference and training to power next-generation ai scale [`UNREAD`]

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

#### Sparsely Activated Mixture-of-Experts are Robust Multi-Task Learners [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2204.07689.pdf)

citation:
```bibtex
@misc{gupta2022sparsely,
      title={Sparsely Activated Mixture-of-Experts are Robust Multi-Task Learners}, 
      author={Shashank Gupta and Subhabrata Mukherjee and Krishan Subudhi and Eduardo Gonzalez and Damien Jose and Ahmed H. Awadallah and Jianfeng Gao},
      year={2022},
      eprint={2204.07689},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

#### Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity (SMoE) [`READ`]

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

#### Beyond distillation: Task-level mixture-of-experts for efficient inference [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2110.03742)

citation:

```bibtex
@article{kudugunta2021beyond,
  title={Beyond distillation: Task-level mixture-of-experts for efficient inference},
  author={Kudugunta, Sneha and Huang, Yanping and Bapna, Ankur and Krikun, Maxim and Lepikhin, Dmitry and Luong, Minh-Thang and Firat, Orhan},
  journal={arXiv preprint arXiv:2110.03742},
  year={2021}
}
```


#### Hash layers for large sparse models [`UNREAD`]

paper link: [here](https://proceedings.neurips.cc/paper_files/paper/2021/file/92bf5e6240737e0326ea59846a83e076-Paper.pdf)

citation:

```bibtex
@article{roller2021hash,
  title={Hash layers for large sparse models},
  author={Roller, Stephen and Sukhbaatar, Sainbayar and Weston, Jason and others},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={17555--17566},
  year={2021}
}
```


#### Sparse is enough in scaling transformers [`UNREAD`]

paper link: [here](https://proceedings.neurips.cc/paper/2021/file/51f15efdd170e6043fa02a74882f0470-Paper.pdf)

citation: 
```bibtex
@article{jaszczur2021sparse,
  title={Sparse is enough in scaling transformers},
  author={Jaszczur, Sebastian and Chowdhery, Aakanksha and Mohiuddin, Afroz and Kaiser, Lukasz and Gajewski, Wojciech and Michalewski, Henryk and Kanerva, Jonni},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={9895--9907},
  year={2021}
}
```


#### Gshard: Scaling giant models with conditional computation and automatic sharding [`UNREAD`]

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

#### Outrageously large neural networks: The sparsely-gated mixture-of-experts layer (Sparse MoE) [`READ`]

paper link: [here](https://arxiv.org/pdf/1701.06538.pdf)

citation:
```bibtex
@article{shazeer2017outrageously,
  title={Outrageously large neural networks: The sparsely-gated mixture-of-experts layer},
  author={Shazeer, Noam and Mirhoseini, Azalia and Maziarz, Krzysztof and Davis, Andy and Le, Quoc and Hinton, Geoffrey and Dean, Jeff},
  journal={arXiv preprint arXiv:1701.06538},
  year={2017}
}
```

#### Adaptive Mixture of Local Experts [`READ`]

paper link: [here](http://www.cs.utoronto.ca/~hinton/absps/jjnh91.ps)

citation:
```bibtex
@article{jacobs1991adaptive,
  title={Adaptive mixtures of local experts},
  author={Jacobs, Robert A and Jordan, Michael I and Nowlan, Steven J and Hinton, Geoffrey E},
  journal={Neural computation},
  volume={3},
  number={1},
  pages={79--87},
  year={1991},
  publisher={MIT Press}
}
```