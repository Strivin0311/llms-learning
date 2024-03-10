# Evaluation on LLMs
*Here're some resources about Evaluation on LLMs*


## Benchmarks

#### GPT-Fathom- Benchmarking Large Language Models to Decipher the Evolutionary Path towards GPT-4 and Beyond ['READ']

paper link: [here](https://arxiv.org/pdf/2309.16583)

citation
```bibtex
@article{zheng2023gpt,
  title={GPT-Fathom: Benchmarking Large Language Models to Decipher the Evolutionary Path towards GPT-4 and Beyond},
  author={Zheng, Shen and Zhang, Yuyu and Zhu, Yijie and Xi, Chenguang and Gao, Pengyang and Zhou, Xun and Chang, Kevin Chen-Chuan},
  journal={arXiv preprint arXiv:2309.16583},
  year={2023}
}
```


#### Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2306.05685.pdf)

homepage link (chatbot Arena): [here](https://chat.lmsys.org/)

citation:
```bibtex
@misc{zheng2023judging,
      title={Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena}, 
      author={Lianmin Zheng and Wei-Lin Chiang and Ying Sheng and Siyuan Zhuang and Zhanghao Wu and Yonghao Zhuang and Zi Lin and Zhuohan Li and Dacheng Li and Eric. P Xing and Hao Zhang and Joseph E. Gonzalez and Ion Stoica},
      year={2023},
      eprint={2306.05685},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### InstructEval: Towards Holistic Evaluation of Instruction-Tuned Large Language Models [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2306.04757.pdf)

citation: 
```bibtex
@misc{chia2023instructeval,
      title={INSTRUCTEVAL: Towards Holistic Evaluation of Instruction-Tuned Large Language Models}, 
      author={Yew Ken Chia and Pengfei Hong and Lidong Bing and Soujanya Poria},
      year={2023},
      eprint={2306.04757},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
    


#### Holistic evaluation of language models (HELM) [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2211.09110.pdf)

citation: 
```bibtex
@article{liang2022holistic,
  title={Holistic evaluation of language models},
  author={Liang, Percy and Bommasani, Rishi and Lee, Tony and Tsipras, Dimitris and Soylu, Dilara and Yasunaga, Michihiro and Zhang, Yian and Narayanan, Deepak and Wu, Yuhuai and Kumar, Ananya and others},
  journal={arXiv preprint arXiv:2211.09110},
  year={2022}
}
```


## English

### Mutli-Domain

#### Openagi: When llm meets domain experts [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2304.04370.pdf)

citation: 
```bibtex
@article{ge2023openagi,
  title={Openagi: When llm meets domain experts},
  author={Ge, Yingqiang and Hua, Wenyue and Ji, Jianchao and Tan, Juntao and Xu, Shuyuan and Zhang, Yongfeng},
  journal={arXiv preprint arXiv:2304.04370},
  year={2023}
}
```


#### Measuring massive multitask language understanding (MMLU) [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2009.03300)

github link: [here](https://github.com/hendrycks/test)

citation: 
```bibtex
@article{hendrycks2020measuring,
  title={Measuring massive multitask language understanding},
  author={Hendrycks, Dan and Burns, Collin and Basart, Steven and Zou, Andy and Mazeika, Mantas and Song, Dawn and Steinhardt, Jacob},
  journal={arXiv preprint arXiv:2009.03300},
  year={2020}
}
```

### Math

#### Let's Verify Step by Step (PRM800K) [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2305.20050.pdf)

github link: [here](https://github.com/openai/prm800k)

citation:
```bibtex
@article{lightman2023lets,
      title={Let's Verify Step by Step}, 
      author={Lightman, Hunter and Kosaraju, Vineet and Burda, Yura and Edwards, Harri and Baker, Bowen and Lee, Teddy and Leike, Jan and Schulman, John and Sutskever, Ilya and Cobbe, Karl},
      journal={arXiv preprint arXiv:2305.20050},
      year={2023}
}
```

#### Measuring Mathematical Problem Solving With the MATH Dataset [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2103.03874.pdf)

github link: [here](https://github.com/hendrycks/math)

citation:
```bibtex
@article{hendrycksmath2021,
  title={Measuring Mathematical Problem Solving With the MATH Dataset},
  author={Dan Hendrycks and Collin Burns and Saurav Kadavath and Akul Arora and Steven Basart and Eric Tang and Dawn Song and Jacob Steinhardt},
  journal={NeurIPS},
  year={2021}
}
```


### Code

#### MultiPL-E: A Scalable and Polyglot Approach to Benchmarking Neural Code Generation [`UNREAD`]

paper link: [here](https://ieeexplore.ieee.org/iel7/32/4359463/10103177.pdf)

github link: [here](https://github.com/nuprl/MultiPL-E)

leaderboard link: [here](https://huggingface.co/spaces/bigcode/bigcode-models-leaderboard)
    
citation: 
```bibtex
@article{cassano2023multipl,
  title={MultiPL-E: a scalable and polyglot approach to benchmarking neural code generation},
  author={Cassano, Federico and Gouwar, John and Nguyen, Daniel and Nguyen, Sydney and Phipps-Costin, Luna and Pinckney, Donald and Yee, Ming-Ho and Zi, Yangtian and Anderson, Carolyn Jane and Feldman, Molly Q and others},
  journal={IEEE Transactions on Software Engineering},
  year={2023},
  publisher={IEEE}
}
```


#### Evaluating Large Language Models Trained on Code (HumanEval) [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2107.03374.pdf)

dataset link: [here](https://huggingface.co/datasets/openai_humaneval)

citation: 
```bibtex
@misc{chen2021evaluating,
      title={Evaluating Large Language Models Trained on Code}, 
      author={Mark Chen and Jerry Tworek and Heewoo Jun and Qiming Yuan and Henrique Ponde de Oliveira Pinto and Jared Kaplan and Harri Edwards and Yuri Burda and Nicholas Joseph and Greg Brockman and Alex Ray and Raul Puri and Gretchen Krueger and Michael Petrov and Heidy Khlaaf and Girish Sastry and Pamela Mishkin and Brooke Chan and Scott Gray and Nick Ryder and Mikhail Pavlov and Alethea Power and Lukasz Kaiser and Mohammad Bavarian and Clemens Winter and Philippe Tillet and Felipe Petroski Such and Dave Cummings and Matthias Plappert and Fotios Chantzis and Elizabeth Barnes and Ariel Herbert-Voss and William Hebgen Guss and Alex Nichol and Alex Paino and Nikolas Tezak and Jie Tang and Igor Babuschkin and Suchir Balaji and Shantanu Jain and William Saunders and Christopher Hesse and Andrew N. Carr and Jan Leike and Josh Achiam and Vedant Misra and Evan Morikawa and Alec Radford and Matthew Knight and Miles Brundage and Mira Murati and Katie Mayer and Peter Welinder and Bob McGrew and Dario Amodei and Sam McCandlish and Ilya Sutskever and Wojciech Zaremba},
      year={2021},
      eprint={2107.03374},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


## Chinese


### Mutli-Domain


#### C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2305.08322)

citation: 
```bibtex
@article{huang2023c,
  title={C-eval: A multi-level multi-discipline chinese evaluation suite for foundation models},
  author={Huang, Yuzhen and Bai, Yuzhuo and Zhu, Zhihao and Zhang, Junlei and Zhang, Jinghan and Su, Tangjun and Liu, Junteng and Lv, Chuancheng and Zhang, Yikai and Lei, Jiayi and others},
  journal={arXiv preprint arXiv:2305.08322},
  year={2023}
}
```
    


### Finance

#### CGCE: A Chinese Generative Chat Evaluation Benchmark for General and Financial Domains [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2305.14471)

citation: 
```bibtex
@article{zhang2023cgce,
  title={CGCE: A Chinese Generative Chat Evaluation Benchmark for General and Financial Domains},
  author={Zhang, Xuanyu and Li, Bingbing and Yang, Qing},
  journal={arXiv preprint arXiv:2305.14471},
  year={2023}
}
```


## Multi-Language





## Metrics

#### Language model evaluation beyond perplexity [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2106.00085)

citation: 
```bibtex
@article{meister2021language,
  title={Language model evaluation beyond perplexity},
  author={Meister, Clara and Cotterell, Ryan},
  journal={arXiv preprint arXiv:2106.00085},
  year={2021}
}
```
    
