# Quantization on LLMs
*Here's some resources about Quantization on LLMs*


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

#### Q8BERT: Quantized 8Bit BERT [`READ`]

paper link: [here](https://arxiv.org/pdf/1910.06188.pdf)

citation:
```bibtex
@inproceedings{Zafrir_2019,
   title={Q8BERT: Quantized 8Bit BERT},
   url={http://dx.doi.org/10.1109/EMC2-NIPS53020.2019.00016},
   DOI={10.1109/emc2-nips53020.2019.00016},
   booktitle={2019 Fifth Workshop on Energy Efficient Machine Learning and Cognitive Computing - NeurIPS Edition (EMC2-NIPS)},
   publisher={IEEE},
   author={Zafrir, Ofir and Boudoukh, Guy and Izsak, Peter and Wasserblat, Moshe},
   year={2019},
   month=dec 
}
```



### Post-Training Quantization (PTQ)

#### HQQ: Half-Quadratic Quantization of Large Machine Learning Models [`UNREAD`]

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
