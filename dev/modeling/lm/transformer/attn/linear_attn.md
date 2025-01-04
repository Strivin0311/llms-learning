
# Linear Attention
*Here're some resources about Linear Attention modules in language modeling*


#### Linear Attention Sequence Parallelism

tag: `LASP` | `OpenNLPLab`

paper link: [here](https://arxiv.org/pdf/2404.02882)

code link: [here](https://github.com/OpenNLPLab/LASP)

citation:

```bibtex
@misc{sun2024linearattentionsequenceparallelism,
      title={Linear Attention Sequence Parallelism}, 
      author={Weigao Sun and Zhen Qin and Dong Li and Xuyang Shen and Yu Qiao and Yiran Zhong},
      year={2024},
      eprint={2404.02882},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
      url={https://arxiv.org/abs/2404.02882}, 
}
```


#### The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry

tag: `Hedgehog` | `Softmax Mimicry` | `ICLR24` | `Stanford University`

paper link: [here](https://openreview.net/pdf?id=4g02l2N2Nx)

citation:

```bibtex
@misc{zhang2024hedgehogporcupineexpressive,
      title={The Hedgehog & the Porcupine: Expressive Linear Attentions with Softmax Mimicry}, 
      author={Michael Zhang and Kush Bhatia and Hermann Kumbong and Christopher Ré},
      year={2024},
      eprint={2402.04347},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2402.04347}, 
}
```


#### Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models

tag: `Lightning Attention 2` | `OpenNLPLab`

paper link: [here](https://arxiv.org/pdf/2401.04658.pdf)

code link: [here](https://github.com/OpenNLPLab/lightning-attention)

follow-up work: [here](https://arxiv.org/pdf/2404.02882)

citation:

```bibtex
@misc{qin2024lightning,
      title={Lightning Attention-2: A Free Lunch for Handling Unlimited Sequence Lengths in Large Language Models}, 
      author={Zhen Qin and Weigao Sun and Dong Li and Xuyang Shen and Weixuan Sun and Yiran Zhong},
      year={2024},
      eprint={2401.04658},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### TransNormerLLM: A Faster and Better Large Language Model with Improved TransNormer

tag: `TransNormerLLM` | `Lightning Attention` | `OpenNLPLab`

paper link: [here](https://arxiv.org/pdf/2307.14995.pdf)

code link: [here](https://github.com/OpenNLPLab/TransnormerLLM)

follow-up work: [here](https://arxiv.org/pdf/2401.04658.pdf)

citation:

```bibtex
@misc{qin2024transnormerllm,
      title={TransNormerLLM: A Faster and Better Large Language Model with Improved TransNormer}, 
      author={Zhen Qin and Dong Li and Weigao Sun and Weixuan Sun and Xuyang Shen and Xiaodong Han and Yunshen Wei and Baohong Lv and Xiao Luo and Yu Qiao and Yiran Zhong},
      year={2024},
      eprint={2307.14995},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### Primal-Attention: Self-attention through Asymmetric Kernel SVD in Primal Representation

tag: `Primal Attention` | `NIPS23`

paper link: [here](https://proceedings.neurips.cc/paper_files/paper/2023/file/cd687a58a13b673eea3fc1b2e4944cf7-Paper-Conference.pdf)

code link: [here](https://github.com/yingyichen-cyy/PrimalAttention)

```bibtex
@misc{chen2023primalattention,
      title={Primal-Attention: Self-attention through Asymmetric Kernel SVD in Primal Representation}, 
      author={Yingyi Chen and Qinghua Tao and Francesco Tonin and Johan A. K. Suykens},
      year={2023},
      eprint={2305.19798},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

#### Fourierformer: Transformer meets generalized fourier integral theorem

tag: `Fourierformer` | `NIPS22` | `UCLA`

paper link: [here](https://proceedings.neurips.cc/paper_files/paper/2022/file/bc968adbdff4a2551649d464b83f264a-Paper-Conference.pdf)

code link: [here](https://github.com/minhtannguyen/FourierFormer_NeurIPS)

citation:

```bibtex
@article{nguyen2022fourierformer,
  title={Fourierformer: Transformer meets generalized fourier integral theorem},
  author={Nguyen, Tan and Pham, Minh and Nguyen, Tam and Nguyen, Khai and Osher, Stanley and Ho, Nhat},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={29319--29335},
  year={2022}
}
```


#### Scatterbrain: Unifying sparse and low-rank attention approximation

tag: `Scatterbrain` | `NIPS21` | `Adobe` | `Stanford University`

overview:

$$
\begin{align}
\widetilde O := \left( \widetilde Q \times \widetilde K^{\mathrm{T}} + S \right)\times V = \widetilde Q \times (\widetilde K^{\mathrm{T}}\times V) + S\times V 
\end{align}
$$

paper link: [here](https://proceedings.neurips.cc/paper/2021/file/9185f3ec501c674c7c788464a36e7fb3-Paper.pdf)

code link: [here](https://github.com/HazyResearch/fly)

citation:

```bibtex
@article{chen2021scatterbrain,
  title={Scatterbrain: Unifying sparse and low-rank attention approximation},
  author={Chen, Beidi and Dao, Tri and Winsor, Eric and Song, Zhao and Rudra, Atri and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:2110.15343},
  year={2021}
}
```


#### Luna: Linear unified nested attention

tag: `Luna` | `NIPS21` | `Meta` | `CMU`

overview:

$$
\begin{align}
&A_{s} := \mathrm{elu}\left( \frac{Q_s \times K^{\mathrm{T}}}{\sqrt{d_k}} \right), \quad\widetilde S := A_{s}\times V, \quad where\quad Q_s := S\times W_q\\
        &A_{u} := \mathrm{softmax}\left( \xi_{\mathbf w_{inv}}(Q, V, A_{s}^{\mathrm{T}}) \right),\;
        \widetilde O := \xi_{\mathbf w_{inv}}(A_{u}, A_{s}^{\mathrm{T}}, V) ,\quad where\quad \mathbf w_{inv} := \left[ i^{-1} \right]_{i=1}^L\
\end{align}
$$

paper link: [here](https://proceedings.neurips.cc/paper/2021/file/14319d9cfc6123106878dc20b94fbaf3-Paper.pdf)

code link: [here](https://github.com/XuezheMax/fairseq-apollo)

citation:

```bibtex
@article{ma2021luna,
  title={Luna: Linear unified nested attention},
  author={Ma, Xuezhe and Kong, Xiang and Wang, Sinong and Zhou, Chunting and May, Jonathan and Ma, Hao and Zettlemoyer, Luke},
  journal={Advances in Neural Information Processing Systems},
  volume={34},
  pages={2441--2453},
  year={2021}
}
```


#### Random feature attention

tag: `RFA` | `ICLR21` | `Google DeepMind`

paper link: [here](https://arxiv.org/pdf/2103.02143)

code link: [here](https://github.com/Noahs-ARK/RFA)

citation:

```bibtex
@article{peng2021random,
  title={Random feature attention},
  author={Peng, Hao and Pappas, Nikolaos and Yogatama, Dani and Schwartz, Roy and Smith, Noah A and Kong, Lingpeng},
  journal={arXiv preprint arXiv:2103.02143},
  year={2021}
}
```


#### Rethinking attention with performers

tag: `Performer` | `ICLR21` | `Google DeepMind`

overview:

$$
\begin{align}
&\mathcal{K_{Pe}}(\mathbf q,\mathbf k) := \mathbb{E_{\omega}}\left[ \varphi_{Pe}(\mathbf q)\times \varphi_{Pe}(\mathbf k){^\mathrm{T}} \right], \\
&where\quad \phi_{Pe}(\mathbf x) = \frac{h(\mathbf x)}{\sqrt{m}} \left[b_1(\omega_1^{\mathrm{T}}\mathbf x),..,b_1(\omega_m^{\mathrm{T}}\mathbf x),.., b_l(\omega_1^{\mathrm{T}}\mathbf x),..,b_l(\omega_m^{\mathrm{T}}\mathbf x) \right]
\end{align}
$$

paper link: [here](https://arxiv.org/pdf/2009.14794)

code link: [here](https://github.com/google-research/google-research/tree/master/performer)

citation:

```bibtex
@article{choromanski2020rethinking,
  title={Rethinking attention with performers},
  author={Choromanski, Krzysztof and Likhosherstov, Valerii and Dohan, David and Song, Xingyou and Gane, Andreea and Sarlos, Tamas and Hawkins, Peter and Davis, Jared and Mohiuddin, Afroz and Kaiser, Lukasz and others},
  journal={arXiv preprint arXiv:2009.14794},
  year={2020}
}
```

#### Linformer: Self-attention with linear complexity

tag: `Linformer` | `Meta`

overview:

$$
\begin{align}
\widetilde O:= \mathrm{softmax}\left( \frac{Q\times\widetilde K^{\mathrm{T}}}{\sqrt{d_k}} \right)\times\widetilde V, \quad\widetilde K = E^{\mathrm{T}} K, \widetilde V = F^{\mathrm{T}} V
\end{align}
$$

paper link: [here](https://arxiv.org/pdf/2006.04768)

code link: [here](https://github.com/facebookresearch/fairseq/blob/main/examples/linformer/README.md)

citation:

```bibtex
@article{wang2020linformer,
  title={Linformer: Self-attention with linear complexity},
  author={Wang, Sinong and Li, Belinda Z and Khabsa, Madian and Fang, Han and Ma, Hao},
  journal={arXiv preprint arXiv:2006.04768},
  year={2020}
}
```


#### Transformers are rnns: Fast autoregressive transformers with linear attention

tag: `Linear Transformer` | `ICML20` | `EPFL`

overview:

$$
\begin{align}
\mathcal{K}_{Li}(\mathbf q,\mathbf k) := \varphi_{Li}(\mathbf q)\times \varphi_{Li}(\mathbf k)^\mathrm{T}, \quad where\quad \varphi_{Li}(\mathbf x) = \mathrm{elu}(\mathbf x) + 1
\end{align}
$$

paper link: [here](http://proceedings.mlr.press/v119/katharopoulos20a/katharopoulos20a.pdf)

code link: [here](https://github.com/lucidrains/linear-attention-transformer)

citation:

```bibtex
@inproceedings{katharopoulos2020transformers,
  title={Transformers are rnns: Fast autoregressive transformers with linear attention},
  author={Katharopoulos, Angelos and Vyas, Apoorv and Pappas, Nikolaos and Fleuret, Fran{\c{c}}ois},
  booktitle={International conference on machine learning},
  pages={5156--5165},
  year={2020},
  organization={PMLR}
}
```
    