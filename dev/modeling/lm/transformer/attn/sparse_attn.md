# Sparse Attention
*Here're some resources about Sparse Attention modules in language modeling*


#### Star Attention: Efficient LLM Inference over Long Sequences

tag: `Star Attention` | `Nvidia`

paper link: [here](https://arxiv.org/pdf/2411.17116)

github link: [here](https://github.com/NVIDIA/Star-Attention)

citation:

```bibtex
@misc{acharya2024starattentionefficientllm,
      title={Star Attention: Efficient LLM Inference over Long Sequences}, 
      author={Shantanu Acharya and Fei Jia and Boris Ginsburg},
      year={2024},
      eprint={2411.17116},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.17116}, 
}
```


#### Post-Training Sparse Attention with Double Sparsity

tag: `Double Sparsity` | `UCB` | `Standford University` | `Shanghai Jiao Tong University`

paper link: [here](https://arxiv.org/pdf/2408.07092)

github link: [here](https://github.com/andy-yang-1/DoubleSparse)

citation:

```bibtex
@misc{yang2024posttrainingsparseattentiondouble,
      title={Post-Training Sparse Attention with Double Sparsity}, 
      author={Shuo Yang and Ying Sheng and Joseph E. Gonzalez and Ion Stoica and Lianmin Zheng},
      year={2024},
      eprint={2408.07092},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2408.07092}, 
}
```


#### MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention

tag: `Dynamic Sparse Attention` | `MInference 1.0` | `NIPS24` | `Microsoft`

paper link: [here](https://arxiv.org/pdf/2407.02490)

github link: [here](https://github.com/microsoft/MInference)

homepage link: [here](https://aka.ms/MInference)

citation:

```bibtex
@misc{jiang2024minference10acceleratingprefilling,
      title={MInference 1.0: Accelerating Pre-filling for Long-Context LLMs via Dynamic Sparse Attention}, 
      author={Huiqiang Jiang and Yucheng Li and Chengruidong Zhang and Qianhui Wu and Xufang Luo and Surin Ahn and Zhenhua Han and Amir H. Abdi and Dongsheng Li and Chin-Yew Lin and Yuqing Yang and Lili Qiu},
      year={2024},
      eprint={2407.02490},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.02490}, 
}
```


#### Faster Neighborhood Attention: Reducing the O(n^2) Cost of Self Attention at the Threadblock Level

tag: `Fused NA` | `Faster Neighborhood Attention` | `NIPS24` | `SHI Labs` | `NVIDIA`

paper link: [here](https://arxiv.org/pdf/2403.04690)

github link: [here](https://github.com/SHI-Labs/NATTEN)

citation:

```bibtex
@misc{hassani2024fasterneighborhoodattentionreducing,
      title={Faster Neighborhood Attention: Reducing the O(n^2) Cost of Self Attention at the Threadblock Level}, 
      author={Ali Hassani and Wen-Mei Hwu and Humphrey Shi},
      year={2024},
      eprint={2403.04690},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2403.04690}, 
}
```


#### LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning

tag: `LongLM` | `ICML24` | `Amazon` | `TAMU`

paper link: [here](https://arxiv.org/pdf/2401.01325.pdf)

github link: [here](https://github.com/datamllab/LongLM)

citation:

```bibtex
@misc{jin2024llm,
      title={LLM Maybe LongLM: Self-Extend LLM Context Window Without Tuning}, 
      author={Hongye Jin and Xiaotian Han and Jingfeng Yang and Zhimeng Jiang and Zirui Liu and Chia-Yuan Chang and Huiyuan Chen and Xia Hu},
      year={2024},
      eprint={2401.01325},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

#### Efficient streaming language models with attention sinks

tag: `Attention Sink` | `StreamingLLM` | `ICLR24` | `Meta` | `MIT`

paper link: [here](https://arxiv.org/pdf/2309.17453)

github link: [here](https://github.com/mit-han-lab/streaming-llm)

citation:

```bibtex
@article{xiao2023efficient,
  title={Efficient streaming language models with attention sinks},
  author={Xiao, Guangxuan and Tian, Yuandong and Chen, Beidi and Han, Song and Lewis, Mike},
  journal={arXiv preprint arXiv:2309.17453},
  year={2023}
}
```


#### LongLoRA: Efficient fine-tuning of long-context large language models

tag: `LongLoRA` | `ICLR24` | `Nvidia` | `MIT` | `CUHK`

paper link: [here](https://arxiv.org/pdf/2309.12307.pdf)

github link: [here](https://github.com/dvlab-research/LongLoRA)

citation:

```bibtex
@article{chen2023longlora,
  title={Longlora: Efficient fine-tuning of long-context large language models},
  author={Chen, Yukang and Qian, Shengju and Tang, Haotian and Lai, Xin and Liu, Zhijian and Han, Song and Jia, Jiaya},
  journal={arXiv preprint arXiv:2309.12307},
  year={2023}
}
```


#### LM-Infinite: Simple on-the-fly length generalization for large language models

tag: `LM-Infinite` | `NAACL24` | `Meta`

paper link: [here](https://aclanthology.org/2024.naacl-long.222.pdf)

github link: [here](https://github.com/Glaciohound/LM-Infinite)

citation:

```bibtex
@article{han2023lm,
  title={Lm-infinite: Simple on-the-fly length generalization for large language models},
  author={Han, Chi and Wang, Qifan and Xiong, Wenhan and Chen, Yu and Ji, Heng and Wang, Sinong},
  journal={arXiv preprint arXiv:2308.16137},
  year={2023}
}
```


#### Longnet: Scaling transformers to 1,000,000,000 tokens

tag: `LongNet` | `Microsoft`

paper link: [here](https://arxiv.org/pdf/2307.02486.pdf)

github link: [here](https://github.com/microsoft/unilm)

citation:

```bibtex
@article{ding2023longnet,
  title={Longnet: Scaling transformers to 1,000,000,000 tokens},
  author={Ding, Jiayu and Ma, Shuming and Dong, Li and Zhang, Xingxing and Huang, Shaohan and Wang, Wenhui and Wei, Furu},
  journal={arXiv preprint arXiv:2307.02486},
  year={2023}
}
```


#### Landmark Attention: Random-Access Infinite Context Length for Transformers

tag: `Landmark Attention` | `NIPS23` | `EPFL`

paper link: [here](https://arxiv.org/pdf/2305.16300)

github link: [here](https://github.com/epfml/landmark-attention/)

citation:

```bibtex
@article{mohtashami2023landmark,
  title={Landmark Attention: Random-Access Infinite Context Length for Transformers},
  author={Mohtashami, Amirkeivan and Jaggi, Martin},
  journal={arXiv preprint arXiv:2305.16300},
  year={2023}
}
```


#### Mixture of Attention Heads: Selecting Attention Heads Per Token

tag: `MoA` | `Mixture of Attention Heads` | `Beihang University`

paper link: [here](https://arxiv.org/pdf/2210.05144)

github link: [here](https://github.com/yikangshen/MoA)

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


#### Mega: Moving Average Equipped Gated Attention

tag: `MEGA` | `ICLR23` | `Meta`

paper link: [here](https://openreview.net/pdf?id=qNLe3iq2El)

github link: [here](https://github.com/facebookresearch/mega)

citation:

```bibtex
@misc{ma2023mega,
      title={Mega: Moving Average Equipped Gated Attention}, 
      author={Xuezhe Ma and Chunting Zhou and Xiang Kong and Junxian He and Liangke Gui and Graham Neubig and Jonathan May and Luke Zettlemoyer},
      year={2023},
      eprint={2209.10655},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### Neighborhood Attention Transformer

tag: `NAT` | `Neighborhood Attention` | `CVPR23` | `SHI Labs` | `Meta`

paper link: [here](https://openaccess.thecvf.com/content/CVPR2023/papers/Hassani_Neighborhood_Attention_Transformer_CVPR_2023_paper.pdf)

github link: [here](https://github.com/SHI-Labs/NATTEN)

follow-up work: [here](https://arxiv.org/pdf/2403.04690)

citation:

```bibtex
@misc{hassani2023neighborhoodattentiontransformer,
      title={Neighborhood Attention Transformer}, 
      author={Ali Hassani and Steven Walton and Jiachen Li and Shen Li and Humphrey Shi},
      year={2023},
      eprint={2204.07143},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2204.07143}, 
}
```


#### Efficient content-based sparse attention with routing transformers

tag: `Routing Transformer` | `TACL21` | `Google`

paper link: [here](https://arxiv.org/pdf/2003.05997)

github link: [here](https://github.com/google-research/google-research/tree/master/routing_transformer)

citation:

```bibtex
@article{roy2021efficient,
  title={Efficient content-based sparse attention with routing transformers},
  author={Roy, Aurko and Saffar, Mohammad and Vaswani, Ashish and Grangier, David},
  journal={Transactions of the Association for Computational Linguistics},
  volume={9},
  pages={53--68},
  year={2021},
  publisher={MIT Press One Rogers Street, Cambridge, MA 02142-1209, USA journals-info~…}
}
```


#### Not all memories are created equal: Learning to forget by expiring

tag: `Expire-Span` | `ICML21` | `Meta`

paper link: [here](http://proceedings.mlr.press/v139/sukhbaatar21a/sukhbaatar21a.pdf)

github link: [here](https://github.com/facebookresearch/transformer-sequential)

citation:

```bibtex
@inproceedings{sukhbaatar2021not,
  title={Not all memories are created equal: Learning to forget by expiring},
  author={Sukhbaatar, Sainbayar and Ju, Da and Poff, Spencer and Roller, Stephen and Szlam, Arthur and Weston, Jason and Fan, Angela},
  booktitle={International Conference on Machine Learning},
  pages={9902--9912},
  year={2021},
  organization={PMLR}
}
```

#### DeepSpeed Sparse Attention: Powering 10x longer sequences with 6x faster execution

tag: `DeepSpeed Sparse Attention` | `DeepSpeed` | `Microsoft`

blog link: [here](https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/)

citation:

```bibtex
@misc{microsoft2020deepspeed,
  author = {Microsoft},
  title = {DeepSpeed Sparse Attention: Powering 10x longer sequences with 6x faster execution},
  year = {2020},
  howpublished = {\url{https://www.microsoft.com/en-us/research/blog/deepspeed-extreme-scale-model-training-for-everyone/}},
}
```


#### Lite transformer with long-short range attention

tag: `Lite Transformer` | `ICLR20` | `MIT` | `Shanghai Jiao Tong University`

paper link: [here](https://arxiv.org/pdf/2004.11886)

github link: [here](https://github.com/mit-han-lab/lite-transformer)

citation:

```bibtex
@article{wu2020lite,
  title={Lite transformer with long-short range attention},
  author={Wu, Zhanghao and Liu, Zhijian and Lin, Ji and Lin, Yujun and Han, Song},
  journal={arXiv preprint arXiv:2004.11886},
  year={2020}
}
```


#### Longformer: The long-document transformer

tag: `Longformer` | `LED` | `Allen AI`

paper link: [here](https://arxiv.org/pdf/2004.05150.pdf)

github link: [here](https://github.com/allenai/longformer)

citation:

```bibtex
@article{beltagy2020longformer,
  title={Longformer: The long-document transformer},
  author={Beltagy, Iz and Peters, Matthew E and Cohan, Arman},
  journal={arXiv preprint arXiv:2004.05150},
  year={2020}
}
```


#### Big bird: Transformers for longer sequences

tag: `Big Bird` | `NIPS20` | `Google`

paper link: [here](https://proceedings.neurips.cc/paper/2020/file/c8512d142a2d849725f31a9a7a361ab9-Paper.pdf)

github link: [here](https://github.com/google-research/bigbird)

citation:

```bibtex
@article{zaheer2020big,
  title={Big bird: Transformers for longer sequences},
  author={Zaheer, Manzil and Guruganesh, Guru and Dubey, Kumar Avinava and Ainslie, Joshua and Alberti, Chris and Ontanon, Santiago and Pham, Philip and Ravula, Anirudh and Wang, Qifan and Yang, Li and others},
  journal={Advances in neural information processing systems},
  volume={33},
  pages={17283--17297},
  year={2020}
}
```


#### Sparse sinkhorn attention

tag: `Sinkhorn` | `ICML20` | `Google`

paper link: [here](https://proceedings.mlr.press/v119/tay20a/tay20a.pdf)

citation:

```bibtex
@misc{tay2020sparsesinkhornattention,
      title={Sparse Sinkhorn Attention}, 
      author={Yi Tay and Dara Bahri and Liu Yang and Donald Metzler and Da-Cheng Juan},
      year={2020},
      eprint={2002.11296},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2002.11296}, 
}
```
    


#### Reformer: The efficient transformer

tag: `Reformer` | `ICLR20` | `Google` | `UC Berkeley`

paper link: [here](https://arxiv.org/pdf/2001.04451)

github link: [here](https://github.com/google/trax/tree/master/trax/models/reformer)

citation:

```bibtex
@article{kitaev2020reformer,
  title={Reformer: The efficient transformer},
  author={Kitaev, Nikita and Kaiser, {\L}ukasz and Levskaya, Anselm},
  journal={arXiv preprint arXiv:2001.04451},
  year={2020}
}
```


#### Blockwise self-attention for long document understanding

tag:  `BlockBERT` | `Blockwise Attention` | `Meta` | `Tsinghua University`

paper link: [here](https://arxiv.org/pdf/1911.02972)

github link: [here](https://github.com/xptree/BlockBERT)

citation:

```bibtex
@article{qiu2019blockwise,
  title={Blockwise self-attention for long document understanding},
  author={Qiu, Jiezhong and Ma, Hao and Levy, Omer and Yih, Scott Wen-tau and Wang, Sinong and Tang, Jie},
  journal={arXiv preprint arXiv:1911.02972},
  year={2019}
}
```


#### Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting

tag: `LogSparse` | `NIPS19` | `UCSB`

paper link: [here](https://proceedings.neurips.cc/paper/2019/file/6775a0635c302542da2c32aa19d86be0-Paper.pdf)

citation:

```bibtex
@article{li2019enhancing,
  title={Enhancing the locality and breaking the memory bottleneck of transformer on time series forecasting},
  author={Li, Shiyang and Jin, Xiaoyong and Xuan, Yao and Zhou, Xiyou and Chen, Wenhu and Wang, Yu-Xiang and Yan, Xifeng},
  journal={Advances in neural information processing systems},
  volume={32},
  year={2019}
}
```

#### Adaptive attention span in transformers

tag: `Adaptive Span` | `Meta`

paper link: [here](https://arxiv.org/pdf/1905.07799)

github link: [here](https://github.com/facebookresearch/adaptive-span)

citation:

```bibtex
@article{sukhbaatar2019adaptive,
  title={Adaptive attention span in transformers},
  author={Sukhbaatar, Sainbayar and Grave, Edouard and Bojanowski, Piotr and Joulin, Armand},
  journal={arXiv preprint arXiv:1905.07799},
  year={2019}
}
```


#### Generating long sequences with sparse transformers

tag: `Sparse Transformer` | `OpenAI`

paper link: [here](https://arxiv.org/pdf/1904.10509)

github link: [here](https://openai.com/index/sparse-transformer/)

citation:

```bibtex
@article{child2019generating,
  title={Generating long sequences with sparse transformers},
  author={Child, Rewon and Gray, Scott and Radford, Alec and Sutskever, Ilya},
  journal={arXiv preprint arXiv:1904.10509},
  year={2019}
}
```

#### Star-transformer

tag: `Star Transformer` | `NAACL19` | `Fudan University`

paper link: [here](https://aclanthology.org/N19-1133.pdf)

github link: [here](https://github.com/dmlc/dgl)

citation:

```bibtex
@article{guo2019star,
  title={Star-transformer},
  author={Guo, Qipeng and Qiu, Xipeng and Liu, Pengfei and Shao, Yunfan and Xue, Xiangyang and Zhang, Zheng},
  journal={arXiv preprint arXiv:1902.09113},
  year={2019}
}
```