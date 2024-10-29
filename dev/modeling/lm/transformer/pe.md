# Position Embedding
*Here're some resources about Position Embedding in Transformers*


#### Effective Long-Context Scaling of Foundation Models

tag: `Llama2 Long`

paper link: [here](https://arxiv.org/pdf/2309.16039.pdf)

citation:

```bibtex
@misc{xiong2023effective,
      title={Effective Long-Context Scaling of Foundation Models}, 
      author={Wenhan Xiong and Jingyu Liu and Igor Molybog and Hejia Zhang and Prajjwal Bhargava and Rui Hou and Louis Martin and Rashi Rungta and Karthik Abinav Sankararaman and Barlas Oguz and Madian Khabsa and Han Fang and Yashar Mehdad and Sharan Narang and Kshitiz Malik and Angela Fan and Shruti Bhosale and Sergey Edunov and Mike Lewis and Sinong Wang and Hao Ma},
      year={2023},
      eprint={2309.16039},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### CoCA: Fusing Position Embedding with Collinear Constrained Attention in Transformers for Long Context Window Extending

tag: `CoCA`

paper link: [here](https://arxiv.org/pdf/2309.08646)

github link: [here](https://github.com/codefuse-ai/Collinear-Constrained-Attention)

citation:

```bibtex
@misc{zhu2024cocafusingpositionembedding,
      title={CoCA: Fusing Position Embedding with Collinear Constrained Attention in Transformers for Long Context Window Extending}, 
      author={Shiyi Zhu and Jing Ye and Wei Jiang and Siqiao Xue and Qi Zhang and Yifan Wu and Jianguo Li},
      year={2024},
      eprint={2309.08646},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2309.08646}, 
}
```


#### ReRoPE for Infinite Extrapolation?

tag: `ReRoPE`

overview:

$$
\begin{align}
    &\quad \widetilde P_{i,j} := \langle R_{\alpha(i,j,w,\kappa)} \mathbf q,\space \mathbf k\rangle, \\
    &\quad where\quad \alpha(i,j,w,\kappa) := \begin{cases}
      \min\lbrace  i-j, w+\frac{i-j-w}{\kappa}\rbrace, & 0<\kappa<\infty\space  (\mathrm{Leaky\space  ReRoPE})\\
      \min\lbrace i-j,w\rbrace & \kappa \rightarrow \infty\space  (\mathrm{ReRoPE})
    \end{cases}
\end{align}
$$

blog link: [here](https://spaces.ac.cn/archives/9708)

citation:

```bibtex
@misc{transformer-upgrade-12,
    author = "Su, Jianlin",
    title = "Transformer Upgrade Roadmap: 12. ReRoPE for Infinite Extrapolation?",
    year = "2023",
    month = "Aug",
    howpublished = "\url{https://spaces.ac.cn/archives/9708}"
}
```


#### YaRN: Efficient Context Window Extension of Large Language Models

tag: `YaRN` | `NTK-by-parts`

paper link: [here](https://arxiv.org/pdf/2309.00071)

github link: [here](https://github.com/jquesnelle/yarn)

citation:

```bibtex
@misc{peng2023yarnefficientcontextwindow,
      title={YaRN: Efficient Context Window Extension of Large Language Models}, 
      author={Bowen Peng and Jeffrey Quesnelle and Honglu Fan and Enrico Shippole},
      year={2023},
      eprint={2309.00071},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2309.00071}, 
}
```


#### Giraffe: Adventures in expanding context lengths in llms

tag: `Giraffe` | `Power-Scaling`

overview:

$$
\begin{align}
  &\quad \widetilde\beta^{i} := \beta^{i} / (1-2i/d)^{\kappa}
\end{align}
$$

paper link: [here](https://arxiv.org/pdf/2308.10882)

citation:

```bibtex
@article{pal2023giraffe,
  title={Giraffe: Adventures in expanding context lengths in llms},
  author={Pal, Arka and Karkhanis, Deep and Roberts, Manley and Dooley, Samuel and Sundararajan, Arvind and Naidu, Siddartha},
  journal={arXiv preprint arXiv:2308.10882},
  year={2023}
}
```


#### NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation

tag: `NTK-aware RoPE` | `NTK-aware Scaled RoPE` | `Dynamic NTK-aware RoPE` | `NTK-mixed RoPE`

overview:

$$
\begin{align}
  &\quad  \widetilde\beta := c_{\kappa}\cdot\beta, \\
  &s.t.\quad\cfrac{n}{\widetilde\beta^{d/2-1}} = \cfrac{n/\kappa}{\beta^{d/2-1}} \Rightarrow c_{\kappa} = \kappa^{2/(d-2)}
\end{align}
$$

blog link: [NTK-Aware Scaled RoPE](https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware_scaled_rope_allows_llama_models_to_have/) | [Dynamic NTK-aware RoPE](https://www.reddit.com/r/LocalLLaMA/comments/14mrgpr/dynamically_scaled_rope_further_increases/) | [NTK-mixed RoPE]([here](https://spaces.ac.cn/archives/9706))

citation:

```bibtex
@misc{ntk-aware-rope,
    author = "bloc97",
    title = "NTK-Aware Scaled RoPE allows LLaMA models to have extended (8k+) context size without any fine-tuning and minimal perplexity degradation",
    year = "2023",
    month = "Jun",
    howpublished = "\url{https://www.reddit.com/r/LocalLLaMA/comments/14lz7j5/ntkaware\\_scaled\\_rope\\_allows\\_llama\\_models\\_to\\_have/}"
}
```


#### Extending context window of large language models via positional interpolation

tag: `PI`

overview:

$$
\begin{align}
  &\quad  \widetilde P_{i,j} := \langle R_{i/\kappa}\mathbf q, R_{j/\kappa}\mathbf k\rangle = \mathbf q^{\mathrm{T}} R_{\frac{j-i}{\kappa}} \mathbf k 
\end{align}
$$

paper link: [here](https://arxiv.org/pdf/2306.15595)

citation:

```bibtex
@article{chen2023extending,
  title={Extending context window of large language models via positional interpolation},
  author={Chen, Shouyuan and Wong, Sherman and Chen, Liangjian and Tian, Yuandong},
  journal={arXiv preprint arXiv:2306.15595},
  year={2023}
}
```


#### A Frustratingly Easy Improvement for Position Embeddings via Random Padding

tag: `Random Padding PE`

paper link: [here](https://arxiv.org/pdf/2305.04859)

citation:

```bibtex
@article{tao2023frustratingly,
  title={A Frustratingly Easy Improvement for Position Embeddings via Random Padding},
  author={Tao, Mingxu and Feng, Yansong and Zhao, Dongyan},
  journal={arXiv preprint arXiv:2305.04859},
  year={2023}
}
```
    
#### Randomized Positional Encodings Boost Length Generalization of Transformers

tag: `Randomized PE`

paper link: [here](https://arxiv.org/pdf/2305.16843)

citation:

```bibtex
@article{ruoss2023randomized,
  title={Randomized Positional Encodings Boost Length Generalization of Transformers},
  author={Ruoss, Anian and Del{\'e}tang, Gr{\'e}goire and Genewein, Tim and Grau-Moya, Jordi and Csord{\'a}s, R{\'o}bert and Bennani, Mehdi and Legg, Shane and Veness, Joel},
  journal={arXiv preprint arXiv:2305.16843},
  year={2023}
}
```


#### A length-extrapolatable transformer

tag: `LEX` | `XPOS`

overview:

$$
  \begin{align}
    & P_{i,j} := \langle\widetilde{\mathbf q_i}, \widetilde{\mathbf k_j} \rangle = \gamma^{i-j}(\mathbf q^{\mathrm{T}} R_{j-i} \mathbf k),\\
    &\quad where\quad \widetilde{\mathbf q_i} := \gamma^i(R_i \mathbf q), \quad \widetilde{\mathbf k_j} := \gamma^{-j} (R_j \mathbf k),\quad i \ge j
  \end{align}
$$

paper link: [here](https://arxiv.org/pdf/2212.10554)

citation:

```bibtex
@article{sun2022length,
  title={A length-extrapolatable transformer},
  author={Sun, Yutao and Dong, Li and Patra, Barun and Ma, Shuming and Huang, Shaohan and Benhaim, Alon and Chaudhary, Vishrav and Song, Xia and Wei, Furu},
  journal={arXiv preprint arXiv:2212.10554},
  year={2022}
}
```


#### SHAPE: Shifted absolute position embedding for transformers

tag: `SHAPE`

paper link: [here](https://arxiv.org/pdf/2109.05644)

citation:

```bibtex
@article{kiyono2021shape,
  title={SHAPE: Shifted absolute position embedding for transformers},
  author={Kiyono, Shun and Kobayashi, Sosuke and Suzuki, Jun and Inui, Kentaro},
  journal={arXiv preprint arXiv:2109.05644},
  year={2021}
}
```


#### Permuteformer: Efficient relative position encoding for long sequences

tag: `Permuteformer`

paper link: [here](https://arxiv.org/pdf/2109.02377)

citation:

```bibtex
@article{chen2021permuteformer,
  title={Permuteformer: Efficient relative position encoding for long sequences},
  author={Chen, Peng},
  journal={arXiv preprint arXiv:2109.02377},
  year={2021}
}
```


#### RoFormer: Enhanced Transformer with Rotary Position Embedding

tag: `RoPE` | `Rotary PE` | `RoFormer`

overview:

$$
\mathrm{RoPE}(n) := \left[
    \begin{matrix}
        R_n^{(0)}\\
        \space  & R_n^{(1)}\\
        \space  & \space  & \ddots\\
        \space  & \space  & \space  & R_n^{(\frac{d}{2}-1)}\\
    \end{matrix}\right], \quad  where\quad  R_n^{(i)} := \left[\begin{matrix}
        \cos(n\theta^i) & -\sin(n\theta^i)\\
        \sin(n\theta^i) & \cos(n\theta^i)\\
    \end{matrix}\right]
$$

paper link: [here](https://arxiv.org/pdf/2104.09864)

blog link: [here](https://huggingface.co/docs/transformers/model_doc/roformer)

citation:

```bibtex
@misc{su2023roformerenhancedtransformerrotary,
      title={RoFormer: Enhanced Transformer with Rotary Position Embedding}, 
      author={Jianlin Su and Yu Lu and Shengfeng Pan and Ahmed Murtadha and Bo Wen and Yunfeng Liu},
      year={2023},
      eprint={2104.09864},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2104.09864}, 
}
```


#### Attention is all you need

tag: `SinPE` | `Sinusoidal PE` | `NIPS17` | `Google`

overview:

$$
\mathrm{SinPE}(n) :=
    \left[\begin{matrix}
        \sin(n\theta^0) \\
        \cos(n\theta^0) \\
        \sin(n\theta^1) \\
        \cos(n\theta^1) \\
        \vdots\\
        \sin(n\theta^{\frac{d}{2}-1})\\
        \cos(n\theta^{\frac{d}{2}-1})\\
    \end{matrix}\right], 
    \quad where\quad  \theta := \beta^{-1}, \space  \beta := base^{\frac{2}{d}}, \space n\in\{0,1,\cdots, L-1\}
$$

paper link: [here](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)

citation:

```bibtex
@misc{vaswani2023attentionneed,
      title={Attention Is All You Need}, 
      author={Ashish Vaswani and Noam Shazeer and Niki Parmar and Jakob Uszkoreit and Llion Jones and Aidan N. Gomez and Lukasz Kaiser and Illia Polosukhin},
      year={2023},
      eprint={1706.03762},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/1706.03762}, 
}
```