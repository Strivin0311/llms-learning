# Alternative DGMs beyond Transformers, Diffusers and AEs
*Here're some resources about Alternative DGMs beyond Transformers, Diffusers and AEs, especially for sequence modeling*


### SSMs


#### Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling

paper link: [here](https://arxiv.org/pdf/2406.07522)

citation:

```bibtex
@misc{ren2024sambasimplehybridstate,
      title={Samba: Simple Hybrid State Space Models for Efficient Unlimited Context Language Modeling}, 
      author={Liliang Ren and Yang Liu and Yadong Lu and Yelong Shen and Chen Liang and Weizhu Chen},
      year={2024},
      eprint={2406.07522},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
      url={https://arxiv.org/abs/2406.07522}, 
}
```

#### Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality

paper link: [here](https://arxiv.org/pdf/2405.21060)

citation:

```bibtex
@misc{dao2024transformersssmsgeneralizedmodels,
      title={Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality}, 
      author={Tri Dao and Albert Gu},
      year={2024},
      eprint={2405.21060},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
      url={https://arxiv.org/abs/2405.21060}, 
}
```


#### Mamba: Linear-Time Sequence Modeling with Selective State Spaces

paper link: [here](https://arxiv.org/abs/2312.00752)

citation: 
```bibtex
@article{gu2023mamba,
  title={Mamba: Linear-Time Sequence Modeling with Selective State Spaces},
  author={Gu, Albert and Dao, Tri},
  journal={arXiv preprint arXiv:2312.00752},
  year={2023}
}
```
    


#### Resurrecting recurrent neural networks for long sequences (LRU)

paper link: [here](https://arxiv.org/pdf/2303.06349)

citation: 
```bibtex
@article{orvieto2023resurrecting,
  title={Resurrecting recurrent neural networks for long sequences},
  author={Orvieto, Antonio and Smith, Samuel L and Gu, Albert and Fernando, Anushan and Gulcehre, Caglar and Pascanu, Razvan and De, Soham},
  journal={arXiv preprint arXiv:2303.06349},
  year={2023}
}
```

#### Effectively modeling time series with simple discrete state spaces (SpaceTime)

paper link: [here](https://arxiv.org/pdf/2303.09489)

citation: 
```bibtex
@article{zhang2023effectively,
  title={Effectively modeling time series with simple discrete state spaces},
  author={Zhang, Michael and Saab, Khaled K and Poli, Michael and Dao, Tri and Goel, Karan and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:2303.09489},
  year={2023}
}
```
    
    


#### Diagonal state space augmented transformers for speech recognition (DSS)

paper link: [here](https://arxiv.org/pdf/2302.14120)

citation: 
```bibtex
@inproceedings{saon2023diagonal,
  title={Diagonal state space augmented transformers for speech recognition},
  author={Saon, George and Gupta, Ankit and Cui, Xiaodong},
  booktitle={ICASSP 2023-2023 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={1--5},
  year={2023},
  organization={IEEE}
}
```
    


#### Simplified state space layers for sequence modeling (S5)

paper link: [here](https://arxiv.org/pdf/2208.04933)

citation: 
```bibtex
@article{smith2022simplified,
  title={Simplified state space layers for sequence modeling},
  author={Smith, Jimmy TH and Warrington, Andrew and Linderman, Scott W},
  journal={arXiv preprint arXiv:2208.04933},
  year={2022}
}
```

#### Long range language modeling via gated state spaces (GSS)

paper link: [here](https://arxiv.org/pdf/2206.13947)

citation: 
```bibtex
@article{mehta2022long,
  title={Long range language modeling via gated state spaces},
  author={Mehta, Harsh and Gupta, Ankit and Cutkosky, Ashok and Neyshabur, Behnam},
  journal={arXiv preprint arXiv:2206.13947},
  year={2022}
}
```
    
    

#### Efficiently modeling long sequences with structured state spaces (S4)

paper link: [here](https://arxiv.org/pdf/2111.00396)

citation: 
```bibtex
@article{gu2021efficiently,
  title={Efficiently modeling long sequences with structured state spaces},
  author={Gu, Albert and Goel, Karan and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:2111.00396},
  year={2021}
}
```


### LongConv


#### Sequence modeling with multiresolution convolutional memory (MultiRes)

paper link: [here](https://proceedings.mlr.press/v202/shi23f/shi23f.pdf)

citation: 
```bibtex
@inproceedings{shi2023sequence,
  title={Sequence modeling with multiresolution convolutional memory},
  author={Shi, Jiaxin and Wang, Ke Alexander and Fox, Emily},
  booktitle={International Conference on Machine Learning},
  pages={31312--31327},
  year={2023},
  organization={PMLR}
}
```
    


#### Hyena hierarchy: Towards larger convolutional language models

paper link: [here](https://arxiv.org/pdf/2302.10866)

citation: 
```bibtex
@article{poli2023hyena,
  title={Hyena hierarchy: Towards larger convolutional language models},
  author={Poli, Michael and Massaroli, Stefano and Nguyen, Eric and Fu, Daniel Y and Dao, Tri and Baccus, Stephen and Bengio, Yoshua and Ermon, Stefano and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:2302.10866},
  year={2023}
}
```

#### Ckconv: Continuous kernel convolution for sequential data

paper link: [here](https://arxiv.org/pdf/2102.02611)

citation: 
```bibtex
@article{romero2021ckconv,
  title={Ckconv: Continuous kernel convolution for sequential data},
  author={Romero, David W and Kuzina, Anna and Bekkers, Erik J and Tomczak, Jakub M and Hoogendoorn, Mark},
  journal={arXiv preprint arXiv:2102.02611},
  year={2021}
}
```
    

### Miscellaneous

#### Scalable MatMul-free Language Modeling

paper link: [here](https://arxiv.org/pdf/2406.02528)

github link: [here](https://github.com/ridgerchu/matmulfreellm)

citation:

```bibtex
@misc{zhu2024scalablematmulfreelanguagemodeling,
      title={Scalable MatMul-free Language Modeling}, 
      author={Rui-Jie Zhu and Yu Zhang and Ethan Sifferman and Tyler Sheaves and Yiqiao Wang and Dustin Richmond and Peng Zhou and Jason K. Eshraghian},
      year={2024},
      eprint={2406.02528},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
      url={https://arxiv.org/abs/2406.02528}, 
}
```


#### Deep equilibrium models

paper link: [here](https://proceedings.neurips.cc/paper/2019/file/01386bd6d8e091c2ab4c7c7de644d37b-Paper.pdf)

citation: 
```bibtex
@article{bai2019deep,
  title={Deep equilibrium models},
  author={Bai, Shaojie and Kolter, J Zico and Koltun, Vladlen},
  journal={Advances in Neural Information Processing Systems},
  volume={32},
  year={2019}
}
```

#### Based: An Educational and Effective Sequence Mixer

blog link: [here](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology2-based)

citation: 
```bibtex
@article{arora2023based,
  title={Based: An Educational and Effective Sequence Mixer},
  author={Simran Arora, Michael Zhang, Sabri Eyuboglu, Chris Ré},
  journal={Hazy Reasearch Blog, Standford},
  year={2023},
  url={https://hazyresearch.stanford.edu/blog/2023-12-11-zoology2-based}
}
```


#### Zoology: Measuring and Improving Recall in Efficient Language Models

paper link: [here](https://arxiv.org/pdf/2312.04927)

blog link: [here](https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis)

citation: 
```bibtex
@article{arora2023zoology,
  title={Zoology: Measuring and Improving Recall in Efficient Language Models},
  author={Arora, Simran and Eyuboglu, Sabri and Timalsina, Aman and Johnson, Isys and Poli, Michael and Zou, James and Rudra, Atri and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:2312.04927},
  year={2023},
  url={https://hazyresearch.stanford.edu/blog/2023-12-11-zoology1-analysis}
}
```

#### Monarch Mixer: A simple sub-quadratic GEMM-based architecture

paper link: [here](https://arxiv.org/pdf/2310.12109)

citation: 
```bibtex
@article{fu2023monarch,
  title={Monarch Mixer: A simple sub-quadratic GEMM-based architecture},
  author={Fu, Daniel Y and Arora, Simran and Grogan, Jessica and Johnson, Isys and Eyuboglu, Sabri and Thomas, Armin W and Spector, Benjamin and Poli, Michael and Rudra, Atri and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:2310.12109},
  year={2023}
}
```

#### Bayesian flow networks

paper link: [here](https://arxiv.org/pdf/2308.07037)

citation: 
```bibtex
@article{graves2023bayesian,
  title={Bayesian flow networks},
  author={Graves, Alex and Srivastava, Rupesh Kumar and Atkinson, Timothy and Gomez, Faustino},
  journal={arXiv preprint arXiv:2308.07037},
  year={2023}
}
```


#### Retentive network: A successor to transformer for large language models

paper link: [here](https://arxiv.org/pdf/2307.08621)

citation: 
```bibtex
@article{sun2023retentive,
  title={Retentive network: A successor to transformer for large language models},
  author={Sun, Yutao and Dong, Li and Huang, Shaohan and Ma, Shuming and Xia, Yuqing and Xue, Jilong and Wang, Jianyong and Wei, Furu},
  journal={arXiv preprint arXiv:2307.08621},
  year={2023}
}
```

