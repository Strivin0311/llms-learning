# Quantization Strategies for LLMs Inference
*Here're some resources about Quantization Strategies for LLMs Inference*


### Quantization-Aware Training (QAT)


#### BitNet a4.8: 4-bit Activations for 1-bit LLMs

tag: `BitNet a4.8` | `Microsoft`

paper link: [here](https://arxiv.org/pdf/2411.04965)

code link: [here](https://github.com/microsoft/bitnet)

homepage link: [here](https://thegenerality.com/agi/)

citation:

```bibtex
@misc{wang2024bitneta484bitactivations,
      title={BitNet a4.8: 4-bit Activations for 1-bit LLMs}, 
      author={Hongyu Wang and Shuming Ma and Furu Wei},
      year={2024},
      eprint={2411.04965},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2411.04965}, 
}
```


#### The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits

tag: `BitNet b1.58` | `Microsoft`

paper link: [here](https://arxiv.org/pdf/2402.17764)

code link: [here](https://github.com/microsoft/bitnet)

homepage link: [here](https://thegenerality.com/agi/)

follow-up work: [here](https://arxiv.org/pdf/2411.04965)

citation:

```bibtex
@misc{ma2024era1bitllmslarge,
      title={The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits}, 
      author={Shuming Ma and Hongyu Wang and Lingxiao Ma and Lei Wang and Wenhui Wang and Shaohan Huang and Li Dong and Ruiping Wang and Jilong Xue and Furu Wei},
      year={2024},
      eprint={2402.17764},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2402.17764}, 
}
```


#### Towards 1-bit Machine Learning Models

tag: `HQQ+` | `Mobius Labs`

blog link: [here](https://mobiusml.github.io/1bit_blog/)

code link: [here](https://github.com/mobiusml/1bit)

citation:

```bibtex
@misc{badri2023hqq+,
      title = {Towards 1-bit Machine Learning Models},
      url = {https://mobiusml.github.io/1bit_blog/},
      author = {Hicham Badri and Appu Shaji},
      month = {March},
      year = {2024}
}
```


#### BitNet: Scaling 1-bit Transformers for Large Language Models

tag: `BitNet` | `BitLinear` | `W1A8` | `Microsoft`

paper link: [here](https://arxiv.org/pdf/2310.11453)

code link: [here](https://github.com/microsoft/bitnet)

homepage link: [here](https://thegenerality.com/agi/)

follow-up work: [here](https://arxiv.org/pdf/2402.17764)

citation:

```bibtex
@misc{wang2023bitnet,
      title={BitNet: Scaling 1-bit Transformers for Large Language Models}, 
      author={Hongyu Wang and Shuming Ma and Li Dong and Shaohan Huang and Huaijie Wang and Lingxiao Ma and Fan Yang and Ruiping Wang and Yi Wu and Furu Wei},
      year={2023},
      eprint={2310.11453},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models

tag: `LoftQ` | `ICLR24` | `Microsoft`

paper link: [here](https://openreview.net/pdf?id=LzPWWPAdY4)

gituhub link: [here](https://github.com/yxli2123/LoftQ)

modelhub link: [here](https://huggingface.co/LoftQ)

citation:

```bibtex
@article{li2023loftq,
  title={Loftq: Lora-fine-tuning-aware quantization for large language models},
  author={Li, Yixiao and Yu, Yifan and Liang, Chen and He, Pengcheng and Karampatziakis, Nikos and Chen, Weizhu and Zhao, Tuo},
  journal={arXiv preprint arXiv:2310.08659},
  year={2023}
}
```


#### QFT: Quantized Full-parameter Tuning of LLMs with Affordable Resources

tag: `QFT` | `UCAS` | `UC Berkeley`

paper link: [here](https://arxiv.org/pdf/2310.07147)

citation:

```bibtex
@misc{li2023qft,
      title={QFT: Quantized Full-parameter Tuning of LLMs with Affordable Resources}, 
      author={Zhikai Li and Xiaoxuan Liu and Banghua Zhu and Zhen Dong and Qingyi Gu and Kurt Keutzer},
      year={2023},
      eprint={2310.07147},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models

tag: `QA-LoRA` | `ICLR24` | `Huawei`

paper link: [here](https://openreview.net/pdf?id=WvFoJccpo8)

code link: [here](https://github.com/yuhuixu1993/qa-lora)

citation:

```bibtex
@article{xu2023qa,
  title={QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models},
  author={Xu, Yuhui and Xie, Lingxi and Gu, Xiaotao and Chen, Xin and Chang, Heng and Zhang, Hengheng and Chen, Zhensu and Zhang, Xiaopeng and Tian, Qi},
  journal={arXiv preprint arXiv:2309.14717},
  year={2023}
}
```


#### SqueezeLLM: Dense-and-Sparse Quantization

tag: `SqueezeLLM` | `ICML24` | `UC Berkeley`

paper link: [here](https://arxiv.org/pdf/2306.07629)

code link: [here](https://github.com/SqueezeAILab/SqueezeLLM)

citation:

```bibtex
@article{kim2023squeezellm,
  title={SqueezeLLM: Dense-and-Sparse Quantization},
  author={Kim, Sehoon and Hooper, Coleman and Gholami, Amir and Dong, Zhen and Li, Xiuyu and Shen, Sheng and Mahoney, Michael W and Keutzer, Kurt},
  journal={arXiv preprint arXiv:2306.07629},
  year={2023}
}
```


#### SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression

tag: `SpQR` | `ICLR24` | `University of Washington`

paper link: [here](https://arxiv.org/pdf/2306.03078)

code link: [here](https://github.com/Vahe1994/SpQR)

citation:

```bibtex
@article{dettmers2023spqr,
  title={SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression},
  author={Dettmers, Tim and Svirschevski, Ruslan and Egiazarian, Vage and Kuznedelev, Denis and Frantar, Elias and Ashkboos, Saleh and Borzunov, Alexander and Hoefler, Torsten and Alistarh, Dan},
  journal={arXiv preprint arXiv:2306.03078},
  year={2023}
}
```

#### Memory-efficient fine-tuning of compressed large language models via sub-4-bit integer quantization

tag: `PEQA` | `NIPS23` | `NAVER Cloud`

paper link: [here](https://proceedings.neurips.cc/paper_files/paper/2023/file/7183f4fc87598f6c6e947b96714acbd6-Paper-Conference.pdf)

citation: 

```bibtex
@article{kim2024memory,
  title={Memory-efficient fine-tuning of compressed large language models via sub-4-bit integer quantization},
  author={Kim, Jeonghoon and Lee, Jung Hyun and Kim, Sungdong and Park, Joonsuk and Yoo, Kang Min and Kwon, Se Jung and Lee, Dongsoo},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}
```


#### Q8BERT: Quantized 8Bit BERT

tag: `Q8BERT` | `NIPS19` | `Intel AI Lab`

paper link: [here](https://arxiv.org/pdf/1910.06188.pdf)

code link: [here](https://github.com/IntelLabs/nlp-architect)

citation:

```bibtex
@inproceedings{zafrir2019q8bert,
    author = "Zafrir, Ofir and Boudoukh, Guy and Izsak, Peter and Wasserblat, Moshe",
    title = "Q8bert: Quantized 8bit bert",
    booktitle = "2019 Fifth Workshop on Energy Efficient Machine Learning and Cognitive Computing-NeurIPS Edition (EMC2-NIPS)",
    pages = "36--39",
    year = "2019",
    organization = "IEEE"
}
```



### Post-Training Quantization (PTQ)


#### DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs

tag: `DuQuant` | `NIPS24` | `UCAS` | `Tsinghua University`

paper link: [here](https://arxiv.org/pdf/2406.01721)

code link: [here](https://github.com/Hsu1023/DuQuant)

citation:

```bibtex
@misc{lin2024duquantdistributingoutliersdual,
      title={DuQuant: Distributing Outliers via Dual Transformation Makes Stronger Quantized LLMs}, 
      author={Haokun Lin and Haobo Xu and Yichen Wu and Jingzhi Cui and Yingtao Zhang and Linzhan Mou and Linqi Song and Zhenan Sun and Ying Wei},
      year={2024},
      eprint={2406.01721},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.01721}, 
}
```


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


#### HQQ: Half-Quadratic Quantization of Large Machine Learning Models

tag: `HQQ` | `Mobius Labs`

blog link: [here](https://mobiusml.github.io/hqq_blog/)

code link: [here](https://github.com/mobiusml/hqq)

follow-up link: [here](https://mobiusml.github.io/1bit_blog/)

citation:

```bibtex
@misc{badri2023hqq,
	title = {Half-Quadratic Quantization of Large Machine Learning Models},
	url = {https://mobiusml.github.io/hqq_blog/},
	author = {Hicham Badri and Appu Shaji},
	month = {November},
	year = {2023}
}
```

#### ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats

tag: `ZeroQuant-FP` | `W4A8` | `NIPS23` | `DeepSpeed` | `Microsoft`

paper link: [here](https://neurips2023-enlsp.github.io/papers/paper_92.pdf)

code link: [here](https://github.com/microsoft/DeepSpeed)

citation:

```bibtex
@article{wu2023zeroquant,
  title={Zeroquant-fp: A leap forward in llms post-training w4a8 quantization using floating-point formats},
  author={Wu, Xiaoxia and Yao, Zhewei and He, Yuxiong},
  journal={arXiv preprint arXiv:2307.09782},
  year={2023}
}
```

#### AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration

tag: `AWQ` | `MLSys24` | `Nvidia` | `MIT` | `Tsinghua University`

paper link: [here](https://proceedings.mlsys.org/paper_files/paper/2024/file/42a452cbafa9dd64e9ba4aa95cc1ef21-Paper-Conference.pdf)

code link: [here](https://github.com/mit-han-lab/llm-awq)

citation:

```bibtex
@misc{lin2023awq,
      title={AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration}, 
      author={Ji Lin and Jiaming Tang and Haotian Tang and Shang Yang and Xingyu Dang and Chuang Gan and Song Han},
      year={2023},
      eprint={2306.00978},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation

tag: `ZeroQuant V2` | `DeepSpeed` | `Microsoft`

paper link: [here](https://arxiv.org/pdf/2303.08302)

code link: [here](https://github.com/microsoft/DeepSpeed)

follow-up work: [here](https://arxiv.org/pdf/2307.09782)

citation:

```bibtex
@misc{yao2023zeroquantv2exploringposttrainingquantization,
      title={ZeroQuant-V2: Exploring Post-training Quantization in LLMs from Comprehensive Study to Low Rank Compensation}, 
      author={Zhewei Yao and Xiaoxia Wu and Cheng Li and Stephen Youn and Yuxiong He},
      year={2023},
      eprint={2303.08302},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2303.08302}, 
}
```


#### SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models

tag: `SmoothQuant` | `ICML23` | `Nvidia`

paper link: [here](https://arxiv.org/pdf/2211.10438.pdf)

code link: [here](https://github.com/mit-han-lab/smoothquant)

citation:

```bibtex
@inproceedings{xiao2023smoothquant,
  title={Smoothquant: Accurate and efficient post-training quantization for large language models},
  author={Xiao, Guangxuan and Lin, Ji and Seznec, Mickael and Wu, Hao and Demouth, Julien and Han, Song},
  booktitle={International Conference on Machine Learning},
  pages={38087--38099},
  year={2023},
  organization={PMLR}
}
```


#### GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers

tag: `GPTQ` | `ICLR22` | `ISTA`

paper link: [here](https://arxiv.org/pdf/2210.17323.pdf)

code link: [here](https://github.com/IST-DASLab/gptq)

citation:

```bibtex
@misc{frantar2023gptq,
      title={GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers}, 
      author={Elias Frantar and Saleh Ashkboos and Torsten Hoefler and Dan Alistarh},
      year={2023},
      eprint={2210.17323},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale

tag: `BitsAndBytes` | `Int8 Quantization` | `NIPS22` | `Meta`

paper link: [here](https://arxiv.org/pdf/2208.07339.pdf)

blog link: [here](https://huggingface.co/blog/hf-bitsandbytes-integration)

code link: [here](https://github.com/bitsandbytes-foundation/bitsandbytes)

citation:

```bibtex
@misc{dettmers2022llmint8,
      title={LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale}, 
      author={Tim Dettmers and Mike Lewis and Younes Belkada and Luke Zettlemoyer},
      year={2022},
      eprint={2208.07339},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### FP8 Quantization: The Power of the Exponent

tag: `FP8 Quantization` | `NIPS22` | `Qualcomm AI`

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


#### ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers

tag: `ZeroQuant` | `NIPS22` | `DeepSpeed` | `Microsoft`

paper link: [here](https://proceedings.neurips.cc/paper_files/paper/2022/file/adf7fa39d65e2983d724ff7da57f00ac-Paper-Conference.pdf)

code link: [here](https://github.com/microsoft/DeepSpeed)

follow-up work: [here](https://arxiv.org/pdf/2303.08302)

citation:

```bibtex
@misc{yao2022zeroquantefficientaffordableposttraining,
      title={ZeroQuant: Efficient and Affordable Post-Training Quantization for Large-Scale Transformers}, 
      author={Zhewei Yao and Reza Yazdani Aminabadi and Minjia Zhang and Xiaoxia Wu and Conglong Li and Yuxiong He},
      year={2022},
      eprint={2206.01861},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2206.01861}, 
}
```

#### Training Deep Neural Networks with Joint Quantization and Pruning of Weights and Activations

tag: `QSparse` | `Joint Quantization and Pruning` | `UCSD`

paper link: [here](https://arxiv.org/pdf/2110.08271)

code link: [here](https://github.com/mlzxy/qsparse)

citation:

```bibtex
@misc{zhang2021trainingdeepneuralnetworks,
      title={Training Deep Neural Networks with Joint Quantization and Pruning of Weights and Activations}, 
      author={Xinyu Zhang and Ian Colbert and Ken Kreutz-Delgado and Srinjoy Das},
      year={2021},
      eprint={2110.08271},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2110.08271}, 
}
```


#### Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation

tag: `Integer Quantization` | `Nvidia`

paper link: [here](https://arxiv.org/pdf/2004.09602)

citation:

```bibtex
@misc{wu2020integerquantizationdeeplearning,
      title={Integer Quantization for Deep Learning Inference: Principles and Empirical Evaluation}, 
      author={Hao Wu and Patrick Judd and Xiaojie Zhang and Mikhail Isaev and Paulius Micikevicius},
      year={2020},
      eprint={2004.09602},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2004.09602}, 
}
```

#### Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets

tag: `STE` | `ICLR19` | `UCLA`

paper link: [here](https://arxiv.org/pdf/1903.05662)

citation:

```bibtex
@misc{yin2019understandingstraightthroughestimatortraining,
      title={Understanding Straight-Through Estimator in Training Activation Quantized Neural Nets}, 
      author={Penghang Yin and Jiancheng Lyu and Shuai Zhang and Stanley Osher and Yingyong Qi and Jack Xin},
      year={2019},
      eprint={1903.05662},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/1903.05662}, 
}
```


#### Quantizing deep convolutional networks for efficient inference: A whitepaper

tag: `Quantization Whitepaper` | `Google`

paper link: [here](https://arxiv.org/pdf/1806.08342)

code link: [here](https://github.com/google/gemmlowp)

citation:

```bibtex
@misc{krishnamoorthi2018quantizing,
      title={Quantizing deep convolutional networks for efficient inference: A whitepaper}, 
      author={Raghuraman Krishnamoorthi},
      year={2018},
      eprint={1806.08342},
      archivePrefix={arXiv},
      primaryClass={id='cs.LG' full_name='Machine Learning' is_active=True alt_name=None in_archive='cs' is_general=False description='Papers on all aspects of machine learning research (supervised, unsupervised, reinforcement learning, bandit problems, and so on) including also robustness, explanation, fairness, and methodology. cs.LG is also an appropriate primary category for applications of machine learning methods.'}
}
```