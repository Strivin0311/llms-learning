# Optimization Algorithms for LLMs Training
*Here're some resources about Optimization Algorithms for LLMs Training*
*Note that most of them are widely-used in general machine learning / deep learning as well*

## Method

#### AGD: an Auto-switchable Optimizer using Stepwise Gradient Difference for Preconditioning Matrix

tag: `AGD` | `DLRover` | `NIPS23` | `AntGroup`

paper link: [here](https://arxiv.org/pdf/2312.01658)

github link: [here](https://github.com/intelligent-machine-learning/dlrover/tree/master/atorch/atorch/optimizers)

citation:

```bibtex
@misc{yue2023agdautoswitchableoptimizerusing,
      title={AGD: an Auto-switchable Optimizer using Stepwise Gradient Difference for Preconditioning Matrix}, 
      author={Yun Yue and Zhiling Ye and Jiadi Jiang and Yongchao Liu and Ke Zhang},
      year={2023},
      eprint={2312.01658},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
      url={https://arxiv.org/abs/2312.01658}, 
}
```


#### FP8-LM: Training FP8 Large Language Models

tag: `FP8-LM` | `FP8 Optimizer`

paper link: [here](https://arxiv.org/pdf/2310.18313.pdf)

github link: [here](https://github.com/Azure/MS-AMP)

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


#### AdaLomo: Low-memory Optimization with Adaptive Learning Rate

tag: `AdaLomo` | `Shanghai AILab` | `Fudan University`

paper link: [here](https://arxiv.org/pdf/2310.10195)

citation:

```bibtex
@misc{lv2023adalomo,
      title={AdaLomo: Low-memory Optimization with Adaptive Learning Rate}, 
      author={Kai Lv and Hang Yan and Qipeng Guo and Haijun Lv and Xipeng Qiu},
      year={2023},
      eprint={2310.10195},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### Full Parameter Fine-tuning for Large Language Models with Limited Resources

tag: `LOMO` | `OpenLMLab` | `Fudan University`

paper link: [here](https://arxiv.org/pdf/2306.09782)

github link: [here](https://github.com/OpenLMLab/LOMO)

citation:

```bibtex
@misc{lv2024parameterfinetuninglargelanguage,
      title={Full Parameter Fine-tuning for Large Language Models with Limited Resources}, 
      author={Kai Lv and Yuqing Yang and Tengxiao Liu and Qinghui Gao and Qipeng Guo and Xipeng Qiu},
      year={2024},
      eprint={2306.09782},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2306.09782}, 
}
```

#### Stable and low-precision training for large-scale vision-language models

tag: `StableAdamW` | `SwitchBlock` | `NIPS23` | `Meta` | `Allen AI` | `LAION`

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


#### 1-bit Adam: Communication Efficient Large-Scale Training with Adam's Convergence Speed

tag: `1-bit Adam` | `Microsoft`

paper link: [here](https://arxiv.org/pdf/2102.02888.pdf)

github link: [here](https://github.com/microsoft/DeepSpeed)

citation:

```bibtex
@misc{tang20211bit,
      title={1-bit Adam: Communication Efficient Large-Scale Training with Adam's Convergence Speed}, 
      author={Hanlin Tang and Shaoduo Gan and Ammar Ahmad Awan and Samyam Rajbhandari and Conglong Li and Xiangru Lian and Ji Liu and Ce Zhang and Yuxiong He},
      year={2021},
      eprint={2102.02888},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### Large Batch Optimization for Deep Learning: Training BERT in 76 minutes

tag: `LAMB` | `BERT` | `ICLR20` | `Google`

paper link: [here](https://arxiv.org/pdf/1904.00962.pdf)

git link: [here](https://github.com/tensorflow/addons/blob/master/tensorflow_addons/optimizers/lamb.py)

citation:

```bibtex
@misc{you2020largebatchoptimizationdeep,
      title={Large Batch Optimization for Deep Learning: Training BERT in 76 minutes}, 
      author={Yang You and Jing Li and Sashank Reddi and Jonathan Hseu and Sanjiv Kumar and Srinadh Bhojanapalli and Xiaodan Song and James Demmel and Kurt Keutzer and Cho-Jui Hsieh},
      year={2020},
      eprint={1904.00962},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1904.00962}, 
}
```


#### Why AdamW matters

tag: `AdamW`

blog link: [here](https://towardsdatascience.com/why-adamw-matters-736223f31b5d)

citation:

```bibtex
@misc{graetz2018whyadamwmatters,
  author = {Fabio M. Graetz},
  title = {Why AdamW matters},
  year = {2018},
  howpublished = {\url{https://towardsdatascience.com/why-adamw-matters-736223f31b5d}},
}
```


#### Adafactor: Adaptive Learning Rates with Sublinear Memory Cost

tag: `Adafactor` | `Google Brain`

paper link: [here](https://arxiv.org/pdf/1804.04235)

citation:

```bibtex
@misc{shazeer2018adafactoradaptivelearningrates,
      title={Adafactor: Adaptive Learning Rates with Sublinear Memory Cost}, 
      author={Noam Shazeer and Mitchell Stern},
      year={2018},
      eprint={1804.04235},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1804.04235}, 
}
```


#### AdamW - Decoupled Weight Decay Regularization

tag: `AdamW` | `SGDW` | `Weight Decay`

paper link: [here](https://arxiv.org/pdf/1711.05101)

git link: [here](https://github.com/loshchil/AdamW-and-SGDW)

citation:

```bibtex
@misc{loshchilov2019decoupled,
      title={Decoupled Weight Decay Regularization}, 
      author={Ilya Loshchilov and Frank Hutter},
      year={2019},
      eprint={1711.05101},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### Incorporating Nesterov Momentum into Adam

tag: `Nadam` | `NAG` | `Adam` | `ICLR16` | `Stanford University`

paper link: [here](https://openreview.net/pdf/OM0jvwB8jIp57ZJjtNEZ.pdf)

citation:

```bibtex
@inproceedings{dozat2016incorporating,
      title = {Incorporating {Nesterov Momentum into Adam}},
      author = {Dozat, Timothy},
      year = {2016},
      booktitle = {Proceedings of the 4th International Conference on Learning Representations},
      pages = {1--4},
}
```


#### Adam: A Method for Stochastic Optimization

tag: `Adam` | `AdaMax` | `OpenAI`

paper link: [here](https://arxiv.org/pdf/1412.6980.pdf)

citation:

```bibtex
@misc{kingma2017adammethodstochasticoptimization,
      title={Adam: A Method for Stochastic Optimization}, 
      author={Diederik P. Kingma and Jimmy Ba},
      year={2017},
      eprint={1412.6980},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1412.6980}, 
}
```


#### NeuralNetworks for Machine Learning Lecture 6a: Overview of mini-batch gradient descent

tag: `RMSProp` | `Hinton`

slides link: [here](https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf)

citation:

```bibtex
@misc{Hinton2012Lecture6a,
  author = {Geoffrey Hinton and Nitish Srivastava and Kevin Swersky},
  title = {Neural Networks for Machine Learning: Lecture 6a-e, Overview of mini-batch gradient descent and advanced optimization methods},
  year = {2012},
  howpublished = {\url{https://www.cs.toronto.edu/~tijmen/csc321/slides/lecture_slides_lec6.pdf}},
  note = {Accessed: 2024-10-27}
}
```


#### ADADELTA: An Adaptive Learning Rate Method

tag: `AdaDelta` | `Google`

paper link: [here](https://arxiv.org/pdf/1212.5701)

citation:

```bibtex
@article{zeiler2012adadelta,
  title={Adadelta: an adaptive learning rate method},
  author={Zeiler, Matthew D},
  journal={arXiv preprint arXiv:1212.5701},
  year={2012}
}
```


#### Adaptive Subgradient Methods for Online Learning and Stochastic Optimization

tag: `Adagrad` | `JMLR11`

paper link: [here](https://www.jmlr.org/papers/volume12/duchi11a/duchi11a.pdf)

citation:

```bibtex
@article{10.5555/1953048.2021068,
      author = {Duchi, John and Hazan, Elad and Singer, Yoram},
      title = {Adaptive Subgradient Methods for Online Learning and Stochastic Optimization},
      year = {2011},
      issue_date = {2/1/2011},
      publisher = {JMLR.org},
      volume = {12},
      number = {null},
      issn = {1532-4435},
      journal = {J. Mach. Learn. Res.},
      month = jul,
      pages = {2121–2159},
      numpages = {39}
}
```


## Survey


#### An overview of gradient descent optimization algorithms

tag: `Gradient Descent Survey` | `GD` | `BGD` | `SGD` | `MBGD` | `Momentum` | `NAG` | `Adagrad` | `Adadelta` | `RMSprop` | `Adam` | `AdaMax` | `Nadam`
 
paper link: [here](https://arxiv.org/pdf/1609.04747)

citation:

```bibtex
@article{ruder2016overview,
  title={An overview of gradient descent optimization algorithms},
  author={Ruder, Sebastian},
  journal={arXiv preprint arXiv:1609.04747},
  year={2016}
}
```


#### Optimization Methods for Large-Scale Machine Learning

tag: `SGD`

paper link: [here](https://arxiv.org/pdf/1606.04838)

citation:

```bibtex
@misc{bottou2018optimizationmethodslargescalemachine,
      title={Optimization Methods for Large-Scale Machine Learning}, 
      author={Léon Bottou and Frank E. Curtis and Jorge Nocedal},
      year={2018},
      eprint={1606.04838},
      archivePrefix={arXiv},
      primaryClass={stat.ML},
      url={https://arxiv.org/abs/1606.04838}, 
}
```