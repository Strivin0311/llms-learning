
# Full Attention
*Here're some resources about Full Attention modules in language modeling*


#### Differential Transformer

tag: `Diff Transformer` | `Diff Attention` | `Microsoft` | `Tsinghua University`

paper link: [here](https://arxiv.org/pdf/2410.05258)

github link: [here](https://github.com/microsoft/unilm/tree/master/Diff-Transformer)

citation:

```bibtex
@misc{ye2024differentialtransformer,
      title={Differential Transformer}, 
      author={Tianzhu Ye and Li Dong and Yuqing Xia and Yutao Sun and Yi Zhu and Gao Huang and Furu Wei},
      year={2024},
      eprint={2410.05258},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2410.05258}, 
}
```

#### SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration

tag: `Sage Attention` | `Tsinghua University`

paper link: [here](https://arxiv.org/pdf/2410.02367v1)

github link: [here](https://github.com/thu-ml/SageAttention)

citation:

```bibtex
@misc{zhang2024sageattentionaccurate8bitattention,
      title={SageAttention: Accurate 8-Bit Attention for Plug-and-play Inference Acceleration}, 
      author={Jintao Zhang and Jia wei and Pengle Zhang and Jun Zhu and Jianfei Chen},
      year={2024},
      eprint={2410.02367},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.02367}, 
}
```


#### FlashMask: Efficient and Rich Mask Extension of FlashAttention

tag: `Flash Mask` | `Flash Attention` | `Baidu`

paper link: [here](https://arxiv.org/pdf/2410.01359v1)

github link: [here](https://github.com/PaddlePaddle/PaddleNLP)

citation:

```bibtex
@misc{wang2024flashmaskefficientrichmask,
      title={FlashMask: Efficient and Rich Mask Extension of FlashAttention}, 
      author={Guoxia Wang and Jinle Zeng and Xiyuan Xiao and Siming Wu and Jiabin Yang and Lujing Zheng and Zeyu Chen and Jiang Bian and Dianhai Yu and Haifeng Wang},
      year={2024},
      eprint={2410.01359},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2410.01359}, 
}
```


#### FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention

tag: `Flex Attention` | `PyTorch` | `Meta`

blog link: [here](https://pytorch.org/blog/flexattention/)

doc link: [here](https://pytorch.org/docs/main/nn.attention.flex_attention.html#module-torch.nn.attention.flex_attention)

github link: [here](https://github.com/pytorch-labs/attention-gym)

citation:

```bibtex
@misc{he2024flexattention,
  author = {Horace He and Driss Guessous and Yanbo Liang and Joy Dong},
  title  = {FlexAttention: The Flexibility of PyTorch with the Performance of FlashAttention},
  month  = {Aug},
  year= {2024},
  url = {https://pytorch.org/blog/flexattention/},
}
```

#### FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision

tag: `Flash Attention 3` | `Colfax Research` | `Meta` | `Nvidia` | `Princeton University`

paper link: [here](https://arxiv.org/pdf/2407.08608)

blog link: [here](https://tridao.me/blog/2024/flash3/)

github link: [here](https://github.com/Dao-AILab/flash-attention/tree/main/hopper)

citation:

```bibtex
@misc{shah2024flashattention3fastaccurateattention,
      title={FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision}, 
      author={Jay Shah and Ganesh Bikshandi and Ying Zhang and Vijay Thakkar and Pradeep Ramani and Tri Dao},
      year={2024},
      eprint={2407.08608},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.08608}, 
}
```

#### Is Flash Attention Stable?

tag: `Flash Attention` | `Meta` | `Harvard University`

paper link: [here](https://arxiv.org/pdf/2405.02803)

citation:

```bibtex
@misc{golden2024flashattentionstable,
      title={Is Flash Attention Stable?}, 
      author={Alicia Golden and Samuel Hsia and Fei Sun and Bilge Acun and Basil Hosmer and Yejin Lee and Zachary DeVito and Jeff Johnson and Gu-Yeon Wei and David Brooks and Carole-Jean Wu},
      year={2024},
      eprint={2405.02803},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
      url={https://arxiv.org/abs/2405.02803}, 
}
```


#### DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training

tag: `LightSeq` | `DistFlashAttn` | `COLM24` | `UC Berkeley`

paper link: [here](https://arxiv.org/pdf/2310.03294)

github link: [here](https://github.com/RulinShao/LightSeq)

citation:

```bibtex
@misc{li2024distflashattn,
      title={DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training}, 
      author={Dacheng Li and Rulin Shao and Anze Xie and Eric P. Xing and Xuezhe Ma and Ion Stoica and Joseph E. Gonzalez and Hao Zhang},
      year={2024},
      eprint={2310.03294},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### Efficient memory management for large language model serving with pagedattention

tag: `Paged Attention` | `vLLM` | `SOSP23` | `UC Berkeley` | `Stanford University`

paper link: [here](https://dl.acm.org/doi/pdf/10.1145/3600006.3613165)

github link: [here](https://github.com/vllm-project/vllm)

citation:

```bibtex
@inproceedings{kwon2023efficient,
      author = {Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and Sheng, Ying and Zheng, Lianmin and Yu, Cody Hao and Gonzalez, Joseph and Zhang, Hao and Stoica, Ion},
      title = {Efficient Memory Management for Large Language Model Serving with PagedAttention},
      year = {2023},
      isbn = {9798400702297},
      publisher = {Association for Computing Machinery},
      address = {New York, NY, USA},
      url = {https://doi.org/10.1145/3600006.3613165},
      doi = {10.1145/3600006.3613165},
      pages = {611–626},
      numpages = {16},
      location = {Koblenz, Germany},
      series = {SOSP '23}
}
```


#### Flashattention-2: Faster attention with better parallelism and work partitioning

tag: `Flash Attention 2` | `ICLR24` | `Princeton University` | `Stanford University`

derivation manuscript link: [here](./fa2_deriv.md)

paper link: [here](https://openreview.net/pdf?id=mZn2Xyh9Ec)

github link: [here](https://github.com/Dao-AILab/flash-attention)

citation:

```bibtex
@article{dao2023flashattention,
  title={Flashattention-2: Faster attention with better parallelism and work partitioning},
  author={Dao, Tri},
  journal={arXiv preprint arXiv:2307.08691},
  year={2023}
}
```


#### Faster Causal Attention Over Large Sequences Through Sparse Flash Attention

tag: `SCFA` | `NIPS23` | `EPFL`

paper link: [here](https://arxiv.org/pdf/2306.01160)

citation:

```bibtex
@article{pagliardini2023faster,
  title={Faster Causal Attention Over Large Sequences Through Sparse Flash Attention},
  author={Pagliardini, Matteo and Paliotta, Daniele and Jaggi, Martin and Fleuret, Fran{\c{c}}ois},
  journal={arXiv preprint arXiv:2306.01160},
  year={2023}
}
```
    

#### Flashattention: Fast and memory-efficient exact attention with io-awareness

tag: `Flash Attention` | `NIPS22` | `Stanford University`

overview:

$$
\begin{align}
  O &:= \mathrm{softmax}\left( \left[\begin{matrix} P^{(1)} & P^{(2)} \end{matrix} \right]  \right) \left[\begin{matrix} V^{(1)} \\ V^{(2)} \end{matrix} \right]\\
  &= \alpha^{(1)} \mathrm{softmax}(P^{(1)}) V^{(1)} + \alpha^{(2)} \mathrm{softmax}(P^{(2)}) V^{(2)}
\end{align}
$$

paper link: [here](https://proceedings.neurips.cc/paper_files/paper/2022/file/67d57c32e20fd0a7a302cb81d36e40d5-Paper-Conference.pdf)

github link: [here](https://github.com/Dao-AILab/flash-attention)

citation:

```bibtex
@article{dao2022flashattention,
  title={Flashattention: Fast and memory-efficient exact attention with io-awareness},
  author={Dao, Tri and Fu, Dan and Ermon, Stefano and Rudra, Atri and R{\'e}, Christopher},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={16344--16359},
  year={2022}
}
```


#### GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints

tag: `GQA` | `Grouped-Query Attention` | `Google`

paper link: [here](https://arxiv.org/pdf/2305.13245)

citation:

```bibtex
@misc{ainslie2023gqatraininggeneralizedmultiquery,
      title={GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints}, 
      author={Joshua Ainslie and James Lee-Thorp and Michiel de Jong and Yury Zemlyanskiy and Federico Lebrón and Sumit Sanghai},
      year={2023},
      eprint={2305.13245},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2305.13245}, 
}
```


#### Self-attention Does Not Need O(n2) Memory

tag: `Online Attention` | `Online Softmax` | `Google`

paper link: [here](https://arxiv.org/pdf/2112.05682)

citation:

```bibtex
@article{rabe2021self,
  title={Self-attention Does Not Need $ O (n\^{} 2) $ Memory},
  author={Rabe, Markus N and Staats, Charles},
  journal={arXiv preprint arXiv:2112.05682},
  year={2021}
}
```

#### Fastformer: Additive Attention Can Be All You Need

tag: `Fastformer` | `MSRA` | `Tsinghua University`

paper link: [here](https://arxiv.org/pdf/2108.09084)

github link: [here](https://github.com/wuch15/Fastformer)

citation:

```bibtex
@misc{wu2021fastformeradditiveattentionneed,
      title={Fastformer: Additive Attention Can Be All You Need}, 
      author={Chuhan Wu and Fangzhao Wu and Tao Qi and Yongfeng Huang and Xing Xie},
      year={2021},
      eprint={2108.09084},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2108.09084}, 
}
```

#### Fast Transformer Decoding: One Write-Head is All You Need

tag: `MQA` | `Multi-Query Attention` | `Google`

paper link: [here](https://arxiv.org/pdf/1911.02150)

citation:

```bibtex
@misc{shazeer2019fasttransformerdecodingwritehead,
      title={Fast Transformer Decoding: One Write-Head is All You Need}, 
      author={Noam Shazeer},
      year={2019},
      eprint={1911.02150},
      archivePrefix={arXiv},
      primaryClass={cs.NE},
      url={https://arxiv.org/abs/1911.02150}, 
}
```

#### Attention is all you need

tag: `MHA` | `Multi-Head Attention` | `Transformer` | `Self-Attention` | `SinPE` | `NIPS17` | `Google`

paper link: [here](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

blog link: [here](https://nlp.seas.harvard.edu/annotated-transformer/)

github link: [here](https://github.com/tensorflow/tensor2tensor)

citation:

```bibtex
@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, A},
  journal={Advances in Neural Information Processing Systems},
  year={2017}
}
```
