# Robustness Abilities of LLMs
*Here're some resources about Robustness Abilities of LLMs, towards adversarial attack, jailbreak, etc*


### Safety Guardrails

#### Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training

paper link: [here](https://arxiv.org/pdf/2407.09121v1)

github link: [here](https://github.com/robustnlp/derta)

citation:

```bibtex
@misc{yuan2024refusefeelunsafeimproving,
      title={Refuse Whenever You Feel Unsafe: Improving Safety in LLMs via Decoupled Refusal Training}, 
      author={Youliang Yuan and Wenxiang Jiao and Wenxuan Wang and Jen-tse Huang and Jiahao Xu and Tian Liang and Pinjia He and Zhaopeng Tu},
      year={2024},
      eprint={2407.09121},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.09121}, 
}
```

#### Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations

paper link: [here](https://arxiv.org/pdf/2312.06674)

github link: [here](https://github.com/meta-llama/PurpleLlama/tree/main/Llama-Guard)

citation:

```bibtex
@misc{inan2023llamaguardllmbasedinputoutput,
      title={Llama Guard: LLM-based Input-Output Safeguard for Human-AI Conversations}, 
      author={Hakan Inan and Kartikeya Upasani and Jianfeng Chi and Rashi Rungta and Krithika Iyer and Yuning Mao and Michael Tontchev and Qing Hu and Brian Fuller and Davide Testuggine and Madian Khabsa},
      year={2023},
      eprint={2312.06674},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2312.06674}, 
}
```


### Hallucinations Mitigation

#### Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps

paper link: [here](https://arxiv.org/pdf/2407.07071v1)

github link: [here](https://github.com/voidism/lookback-lens)

citation:

```bibtex
@misc{chuang2024lookbacklensdetectingmitigating,
      title={Lookback Lens: Detecting and Mitigating Contextual Hallucinations in Large Language Models Using Only Attention Maps}, 
      author={Yung-Sung Chuang and Linlu Qiu and Cheng-Yu Hsieh and Ranjay Krishna and Yoon Kim and James Glass},
      year={2024},
      eprint={2407.07071},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.07071}, 
}
```

#### Weakly Supervised Detection of Hallucinations in LLM Activations

paper link: [here](https://arxiv.org/pdf/2312.02798.pdf)

citation:

```bibtex
@misc{rateike2023weakly,
      title={Weakly Supervised Detection of Hallucinations in LLM Activations}, 
      author={Miriam Rateike and Celia Cintas and John Wamburu and Tanya Akumu and Skyler Speakman},
      year={2023},
      eprint={2312.02798},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

#### Can LLM-Generated Misinformation Be Detected?

paper link: [here](https://arxiv.org/pdf/2309.13788.pdf)

citation:

```bibtex
@misc{chen2023llmgenerated,
      title={Can LLM-Generated Misinformation Be Detected?}, 
      author={Canyu Chen and Kai Shu},
      year={2023},
      eprint={2309.13788},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

### Adversarial Attack&Defence

#### Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks

paper link: [here](https://arxiv.org/pdf/2401.17263.pdf)

citation:

```bibtex
@misc{zhou2024robust,
      title={Robust Prompt Optimization for Defending Language Models Against Jailbreaking Attacks}, 
      author={Andy Zhou and Bo Li and Haohan Wang},
      year={2024},
      eprint={2401.17263},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

#### Defending Large Language Models Against Jailbreaking Attacks Through Goal Prioritization

paper link: [here](https://arxiv.org/pdf/2311.09096.pdf)

citation:

```bibtex
@misc{zhang2023defending,
      title={Defending Large Language Models Against Jailbreaking Attacks Through Goal Prioritization}, 
      author={Zhexin Zhang and Junxiao Yang and Pei Ke and Minlie Huang},
      year={2023},
      eprint={2311.09096},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

#### Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation

paper link: [here](https://arxiv.org/pdf/2311.03348.pdf)

citation:

```bibtex
@misc{shah2023scalable,
      title={Scalable and Transferable Black-Box Jailbreaks for Language Models via Persona Modulation}, 
      author={Rusheb Shah and Quentin Feuillade--Montixi and Soroush Pour and Arush Tagade and Stephen Casper and Javier Rando},
      year={2023},
      eprint={2311.03348},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game

paper link: [here](https://arxiv.org/pdf/2311.01011.pdf)

citation:

```bibtex
@misc{toyer2023tensor,
      title={Tensor Trust: Interpretable Prompt Injection Attacks from an Online Game}, 
      author={Sam Toyer and Olivia Watkins and Ethan Adrian Mendes and Justin Svegliato and Luke Bailey and Tiffany Wang and Isaac Ong and Karim Elmaaroufi and Pieter Abbeel and Trevor Darrell and Alan Ritter and Stuart Russell},
      year={2023},
      eprint={2311.01011},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### Break it, Imitate it, Fix it: Robustness by Generating Human-Like Attacks

paper link: [here](https://arxiv.org/pdf/2310.16955.pdf)

citation:

```bibtex
@misc{sinha2024break,
      title={Break it, Imitate it, Fix it: Robustness by Generating Human-Like Attacks}, 
      author={Aradhana Sinha and Ananth Balashankar and Ahmad Beirami and Thi Avrahami and Jilin Chen and Alex Beutel},
      year={2024},
      eprint={2310.16955},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### AutoDAN: Interpretable Gradient-Based Adversarial Attacks on Large Language Models

paper link: [here](https://arxiv.org/pdf/2310.15140.pdf)

citation:

```bibtex
@misc{zhu2023autodan,
      title={AutoDAN: Interpretable Gradient-Based Adversarial Attacks on Large Language Models}, 
      author={Sicheng Zhu and Ruiyi Zhang and Bang An and Gang Wu and Joe Barrow and Zichao Wang and Furong Huang and Ani Nenkova and Tong Sun},
      year={2023},
      eprint={2310.15140},
      archivePrefix={arXiv},
      primaryClass={cs.CR}
}
```


#### Jailbreaking Black Box Large Language Models in Twenty Queries

paper link: [here](https://arxiv.org/pdf/2310.08419.pdf)

citation:

```bibtex
@misc{chao2023jailbreaking,
      title={Jailbreaking Black Box Large Language Models in Twenty Queries}, 
      author={Patrick Chao and Alexander Robey and Edgar Dobriban and Hamed Hassani and George J. Pappas and Eric Wong},
      year={2023},
      eprint={2310.08419},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

#### SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks

paper link: [here](https://arxiv.org/pdf/2310.03684.pdf)

citation:

```bibtex
@misc{robey2023smoothllm,
      title={SmoothLLM: Defending Large Language Models Against Jailbreaking Attacks}, 
      author={Alexander Robey and Eric Wong and Hamed Hassani and George J. Pappas},
      year={2023},
      eprint={2310.03684},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

#### Low-Resource Languages Jailbreak GPT-4

paper link: [here](https://arxiv.org/pdf/2310.02446.pdf)

citation:

```bibtex
@misc{yong2024lowresource,
      title={Low-Resource Languages Jailbreak GPT-4}, 
      author={Zheng-Xin Yong and Cristina Menghini and Stephen H. Bach},
      year={2024},
      eprint={2310.02446},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### Baseline Defenses for Adversarial Attacks Against Aligned Language Models

paper link: [here](https://arxiv.org/pdf/2309.00614.pdf)

citation:

```bibtex
@misc{jain2023baseline,
      title={Baseline Defenses for Adversarial Attacks Against Aligned Language Models}, 
      author={Neel Jain and Avi Schwarzschild and Yuxin Wen and Gowthami Somepalli and John Kirchenbauer and Ping-yeh Chiang and Micah Goldblum and Aniruddha Saha and Jonas Geiping and Tom Goldstein},
      year={2023},
      eprint={2309.00614},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### Detecting language model attacks with perplexity

paper link: [here](https://arxiv.org/pdf/2308.14132.pdf)

citation:

```bibtex
@misc{alon2023detecting,
      title={Detecting Language Model Attacks with Perplexity}, 
      author={Gabriel Alon and Michael Kamfonas},
      year={2023},
      eprint={2308.14132},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### Universal and Transferable Adversarial Attacks on Aligned Language Models

paper link: [here](https://arxiv.org/pdf/2307.15043.pdf)

citation:

```bibtex
@misc{zou2023universal,
      title={Universal and Transferable Adversarial Attacks on Aligned Language Models}, 
      author={Andy Zou and Zifan Wang and Nicholas Carlini and Milad Nasr and J. Zico Kolter and Matt Fredrikson},
      year={2023},
      eprint={2307.15043},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### Jailbroken: How does llm safety training fail?

paper link: [here](https://arxiv.org/pdf/2307.02483.pdf)

citation:

```bibtex
@misc{wei2023jailbroken,
      title={Jailbroken: How Does LLM Safety Training Fail?}, 
      author={Alexander Wei and Nika Haghtalab and Jacob Steinhardt},
      year={2023},
      eprint={2307.02483},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```



#### Tricking LLMs into Disobedience: Formalizing, Analyzing, and Detecting Jailbreaks

paper link: [here](https://arxiv.org/pdf/2305.14965.pdf)

citation:

```bibtex
@misc{rao2024tricking,
      title={Tricking LLMs into Disobedience: Formalizing, Analyzing, and Detecting Jailbreaks}, 
      author={Abhinav Rao and Sachin Vashistha and Atharva Naik and Somak Aditya and Monojit Choudhury},
      year={2024},
      eprint={2305.14965},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### Jailbreaking ChatGPT via Prompt Engineering: An Empirical Study

paper link: [here](https://arxiv.org/pdf/2305.13860.pdf)

citation:

```bibtex
@misc{liu2023jailbreaking,
      title={Jailbreaking ChatGPT via Prompt Engineering: An Empirical Study}, 
      author={Yi Liu and Gelei Deng and Zhengzi Xu and Yuekang Li and Yaowen Zheng and Ying Zhang and Lida Zhao and Tianwei Zhang and Yang Liu},
      year={2023},
      eprint={2305.13860},
      archivePrefix={arXiv},
      primaryClass={cs.SE}
}
```












