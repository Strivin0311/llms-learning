
# Full Attention
*Here're some resources about Full Attention modules in language modeling*


#### FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision

tag: `Flash Attention 3`

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

tag: `Flash Attention`

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


#### Flashattention-2: Faster attention with better parallelism and work partitioning

tag: `Flash Attention 2`

derivation manuscript link: [here](./fa2_deriv.md)

paper link: [here](https://arxiv.org/pdf/2307.08691.pdf)

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

tag: `SCFA`

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

tag: `Flash Attention`

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


#### DISTFLASHATTN: Distributed Memory-efficient Attention for Long-context LLMs Training

tag: `LightSeq`

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

tag: `Paged Attention`

paper link: [here](https://arxiv.org/pdf/2309.06180)

github link: [here](https://github.com/vllm-project/vllm)

citation:

```bibtex
@article{kwon2023efficient,
  title={Efficient memory management for large language model serving with pagedattention},
  author={Kwon, Woosuk and Li, Zhuohan and Zhuang, Siyuan and Sheng, Ying and Zheng, Lianmin and Yu, Cody Hao and Gonzalez, Joseph E and Zhang, Hao and Stoica, Ion},
  journal={arXiv preprint arXiv:2309.06180},
  year={2023}
}
```


#### Self-attention Does Not Need Memory

tag: `Online Softmax`

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
