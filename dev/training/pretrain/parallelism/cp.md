# Context Parallelism for LLMs Training
*Here're some resources about Context Parallelism for LLMs Training*
*Note that the "sequence parallelism" usually refers to one attached parallelism strategy along with tensor parallelism, different from context parallelism*


#### Training Variable Sequences with Data-Centric Parallel

tag: `DCP` | `VideoSys` | `NUS`

docs link: [here](https://github.com/NUS-HPC-AI-Lab/VideoSys/blob/master/docs/dcp.md)

code link: [here](https://github.com/NUS-HPC-AI-Lab/VideoSys)

homepage link: [here](https://oahzxl.github.io/DCP/)

citation:

```bibtex
@misc{zhang2024dcp,
  title={Training Variable Sequences with Data-Centric Parallel},
  author={Geng Zhang and Xuanlei Zhao and Kai Wang and Yang You},
  year={2024},
}
```


#### TokenRing: An Efficient Parallelism Framework for Infinite-Context LLMs via Bidirectional Communication

tag: `TokenRing` | `SJTU`

paper link: [here](https://arxiv.org/pdf/2412.20501)

code link: [here](https://github.com/ACA-Lab-SJTU/token-ring)

citation:

```bibtex
@misc{wang2024tokenringefficientparallelismframework,
      title={TokenRing: An Efficient Parallelism Framework for Infinite-Context LLMs via Bidirectional Communication}, 
      author={Zongwu Wang and Fangxin Liu and Mingshuai Li and Li Jiang},
      year={2024},
      eprint={2412.20501},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2412.20501}, 
}
```


#### Data-Centric and Heterogeneity-Adaptive Sequence Parallelism for Efficient LLM Training

tag: `FlexSP` | `Peking University`

paper link: [here](https://arxiv.org/pdf/2412.01523)

citation:

```bibtex
@misc{wang2024datacentricheterogeneityadaptivesequenceparallelism,
      title={Data-Centric and Heterogeneity-Adaptive Sequence Parallelism for Efficient LLM Training}, 
      author={Yujie Wang and Shiju Wang and Shenhan Zhu and Fangcheng Fu and Xinyi Liu and Xuefeng Xiao and Huixia Li and Jiashi Li and Faming Wu and Bin Cui},
      year={2024},
      eprint={2412.01523},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2412.01523}, 
}
```


#### Training Ultra Long Context Language Model with Fully Pipelined Distributed Transformer

tag: `FPDT` | `DeepSpeed Ulysses` | `ZERO-3` | `Microsoft`

paper link: [here](https://arxiv.org/pdf/2408.16978)

code link: [here](https://github.com/microsoft/DeepSpeed/pull/6462)

citation:

```bibtex
@misc{yao2024trainingultralongcontext,
      title={Training Ultra Long Context Language Model with Fully Pipelined Distributed Transformer}, 
      author={Jinghan Yao and Sam Ade Jacobs and Masahiro Tanaka and Olatunji Ruwase and Aamir Shafi and Hari Subramoni and Dhabaleswar K. Panda},
      year={2024},
      eprint={2408.16978},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2408.16978}, 
}
```


#### LongVILA: Scaling Long-Context Visual Language Models for Long Videos

tag: `LongVILA` | `MM-SP` | `NVIDIA` | `MIT` | `UC Berkeley`

paper link: [here](https://arxiv.org/pdf/2408.10188)

code link: [here](https://github.com/NVlabs/VILA/tree/main/longvila)

citation:

```bibtex
@misc{chen2024longvilascalinglongcontextvisual,
      title={LongVILA: Scaling Long-Context Visual Language Models for Long Videos}, 
      author={Yukang Chen and Fuzhao Xue and Dacheng Li and Qinghao Hu and Ligeng Zhu and Xiuyu Li and Yunhao Fang and Haotian Tang and Shang Yang and Zhijian Liu and Ethan He and Hongxu Yin and Pavlo Molchanov and Jan Kautz and Linxi Fan and Yuke Zhu and Yao Lu and Song Han},
      year={2024},
      eprint={2408.10188},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2408.10188}, 
}
```


#### WallFacer: Harnessing Multi-dimensional Ring Parallelism for Efficient Long Sequence Model Training

tag: `WallFacer` | `Ring Attention` | `NUS` | `UC Berkeley`

paper link: [here](https://arxiv.org/pdf/2407.00611)

citation:

```bibtex
@misc{liu2024wallfacerharnessingmultidimensionalring,
      title={WallFacer: Harnessing Multi-dimensional Ring Parallelism for Efficient Long Sequence Model Training}, 
      author={Ziming Liu and Shaoyu Wang and Shenggan Cheng and Zhongkai Zhao and Kai Wang and Xuanlei Zhao and James Demmel and Yang You},
      year={2024},
      eprint={2407.00611},
      archivePrefix={arXiv},
      primaryClass={cs.DC},
      url={https://arxiv.org/abs/2407.00611}, 
}
```


#### USP: A Unified Sequence Parallelism Approach for Long Context Generative AI

tag: `USP` | `Tencent`

paper link: [here](https://arxiv.org/pdf/2405.07719)

code link: [here](https://github.com/feifeibear/long-context-attention)

citation:

```bibtex
@misc{fang2024uspunifiedsequenceparallelism,
      title={USP: A Unified Sequence Parallelism Approach for Long Context Generative AI}, 
      author={Jiarui Fang and Shangchun Zhao},
      year={2024},
      eprint={2405.07719},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2405.07719}, 
}
```

#### Striped Attention: Faster Ring Attention for Causal Transformers

tag: `Striped Attention` | `Ring Attention` | `Load Balance` | `MIT`

paper link: [here](https://arxiv.org/pdf/2311.09431)

code link: [here](https://github.com/exists-forall/striped_attention)

citation:

```bibtex
@misc{brandon2023stripedattentionfasterring,
      title={Striped Attention: Faster Ring Attention for Causal Transformers}, 
      author={William Brandon and Aniruddha Nrusimha and Kevin Qian and Zachary Ankner and Tian Jin and Zhiye Song and Jonathan Ragan-Kelley},
      year={2023},
      eprint={2311.09431},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2311.09431}, 
}
```

#### DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training

tag: `DISTFLASHATTN` | `COLM24` | `UC Berkeley`

paper link: [here](https://arxiv.org/pdf/2310.03294)

code link: [here](https://github.com/RulinShao/LightSeq)

citation:

```bibtex
@misc{li2024distflashattndistributedmemoryefficientattention,
      title={DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training}, 
      author={Dacheng Li and Rulin Shao and Anze Xie and Eric P. Xing and Xuezhe Ma and Ion Stoica and Joseph E. Gonzalez and Hao Zhang},
      year={2024},
      eprint={2310.03294},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2310.03294}, 
}
```


#### Ring Attention with Blockwise Transformers for Near-Infinite Context

tag: `Ring Attention` | `Ring Flash Attention` | `ICLR24` | `UC Berkeley`

paper link: [here](https://arxiv.org/pdf/2310.01889)

code link: [here](https://github.com/haoliuhl/ringattention)

citation:

```bibtex
@misc{liu2023ringattentionblockwisetransformers,
      title={Ring Attention with Blockwise Transformers for Near-Infinite Context}, 
      author={Hao Liu and Matei Zaharia and Pieter Abbeel},
      year={2023},
      eprint={2310.01889},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.01889}, 
}
```


#### System Optimizations for Enabling Training of Extreme Long Sequence Transformer Models

tag: `DeepSpeed Ulysses` | `PODC24` | `Microsoft`

paper link: [here](https://dl.acm.org/doi/pdf/10.1145/3662158.3662806)

blog link: [here](https://github.com/microsoft/DeepSpeed/blob/master/blogs/deepspeed-ulysses/chinese/README.md)

code link: [here](https://github.com/microsoft/DeepSpeed)

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

#### Sequence parallelism: Long sequence training from system perspective

tag: `RSA` | `Ring Self-Attention` | `ACL23` | `NUS`

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


