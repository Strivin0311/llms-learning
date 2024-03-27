# Quantization on LLMs
*Here're some resources about Quantization on LLMs*


## Post-Quantization

#### LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2208.07339.pdf)

blog link: [here](https://huggingface.co/blog/hf-bitsandbytes-integration)

github link: [here](https://github.com/TimDettmers/bitsandbytes)

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

#### 8-bit Optimizers via Block-wise Quantization [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2110.02861.pdf)

github link: [here](https://github.com/TimDettmers/bitsandbytes)

citation:
```bibtex
@misc{dettmers20228bit,
      title={8-bit Optimizers via Block-wise Quantization}, 
      author={Tim Dettmers and Mike Lewis and Sam Shleifer and Luke Zettlemoyer},
      year={2022},
      eprint={2110.02861},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

## Pre-Quantization 

### Quantization-Aware Training (QAT)


#### LoftQ: LoRA-Fine-Tuning-Aware Quantization for Large Language Models [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2310.08659.pdf)

citation:
```bibtex
@article{li2023loftq,
  title={Loftq: Lora-fine-tuning-aware quantization for large language models},
  author={Li, Yixiao and Yu, Yifan and Liang, Chen and He, Pengcheng and Karampatziakis, Nikos and Chen, Weizhu and Zhao, Tuo},
  journal={arXiv preprint arXiv:2310.08659},
  year={2023}
}
```


#### QFT: Quantized Full-parameter Tuning of LLMs with Affordable Resources [`UNREAD`]

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


#### QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2309.14717)

citation: 
```bibtex
@article{xu2023qa,
  title={QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models},
  author={Xu, Yuhui and Xie, Lingxi and Gu, Xiaotao and Chen, Xin and Chang, Heng and Zhang, Hengheng and Chen, Zhensu and Zhang, Xiaopeng and Tian, Qi},
  journal={arXiv preprint arXiv:2309.14717},
  year={2023}
}
```


#### SqueezeLLM: Dense-and-Sparse Quantization [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2306.07629)

citation:
```bibtex
@article{kim2023squeezellm,
  title={SqueezeLLM: Dense-and-Sparse Quantization},
  author={Kim, Sehoon and Hooper, Coleman and Gholami, Amir and Dong, Zhen and Li, Xiuyu and Shen, Sheng and Mahoney, Michael W and Keutzer, Kurt},
  journal={arXiv preprint arXiv:2306.07629},
  year={2023}
}
```


#### SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2306.03078)

citation:
```bibtex
@article{dettmers2023spqr,
  title={SpQR: A Sparse-Quantized Representation for Near-Lossless LLM Weight Compression},
  author={Dettmers, Tim and Svirschevski, Ruslan and Egiazarian, Vage and Kuznedelev, Denis and Frantar, Elias and Ashkboos, Saleh and Borzunov, Alexander and Hoefler, Torsten and Alistarh, Dan},
  journal={arXiv preprint arXiv:2306.03078},
  year={2023}
}
```

#### Memory-efficient fine-tuning of compressed large language models via sub-4-bit integer quantization [`READ`]

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


#### Qlora: Efficient finetuning of quantized llms [`READ`]

paper link: [here](https://arxiv.org/pdf/2305.14314)

github link: [here](https://github.com/artidoro/qlora)

tutorial links:

|tutorial name|public date|main-lib version|notebook link|
|-|-|-|-|
|tutorial_qlora|2024.01|bitsandbytes=0.41.3, peft=0.7.1|[here](../notebooks/tutorial_qlora.ipynb)|

citation: 
```bibtex
@article{dettmers2023qlora,
  title={Qlora: Efficient finetuning of quantized llms},
  author={Dettmers, Tim and Pagnoni, Artidoro and Holtzman, Ari and Zettlemoyer, Luke},
  journal={arXiv preprint arXiv:2305.14314},
  year={2023}
}
```

#### 8-bit Optimizers via Block-wise Quantization [`READ`]

paper link: [here](https://arxiv.org/pdf/2110.02861)

citation:
```bibtex
@misc{dettmers20228bit,
      title={8-bit Optimizers via Block-wise Quantization}, 
      author={Tim Dettmers and Mike Lewis and Sam Shleifer and Luke Zettlemoyer},
      year={2022},
      eprint={2110.02861},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### Q8BERT: Quantized 8Bit BERT [`READ`]

paper link: [here](https://arxiv.org/pdf/1910.06188.pdf)

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

#### HQQ: Half-Quadratic Quantization of Large Machine Learning Models [`READ`]

blog link: [here](https://mobiusml.github.io/hqq_blog/)

github link: [here](https://github.com/mobiusml/hqq)

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

#### ZeroQuant-FP: A Leap Forward in LLMs Post-Training W4A8 Quantization Using Floating-Point Formats [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2307.09782)

citation:
```bibtex
@article{wu2023zeroquant,
  title={Zeroquant-fp: A leap forward in llms post-training w4a8 quantization using floating-point formats},
  author={Wu, Xiaoxia and Yao, Zhewei and He, Yuxiong},
  journal={arXiv preprint arXiv:2307.09782},
  year={2023}
}
```

#### AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2306.00978.pdf)

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

#### GGUF: GPT-Generated Unified Format (llama.cpp) [`READ`]

github link: [here](https://github.com/ggerganov/llama.cpp)

citation:
```bibtex
@misc{gguf,
    author = {Georgi Gerganov},
    title = {GGML: GPT-Generated Model Language},
    year = {2023},
    month = {Aug},
    url = {\url{https://github.com/ggerganov/llama.cpp}},
}
```

#### SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2211.10438.pdf)

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


#### GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2210.17323.pdf)

github link: [here](https://github.com/IST-DASLab/gptq)

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

#### GGML: GPT-Generated Model Language [`READ`]

github link: [here](https://github.com/ggerganov/ggml)

citation:
```bibtex
@misc{ggml,
    author = {Georgi Gerganov},
    title = {GGML: GPT-Generated Model Language},
    year = {2022},
    url = {\url{https://github.com/ggerganov/ggml}},
}
```

#### FP8 Quantization: The Power of the Exponent [`READ`]

paper link: [here](https://arxiv.org/pdf/2208.09225.pdf)

github link: [here](https://github.com/Qualcomm-AI-research/FP8-quantization)

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
