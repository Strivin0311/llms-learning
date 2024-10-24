# Efficient Fine-Tuning
*Here're some resources about Efficient Fine-Tuning strategies for LLMs*
 

### Full-Parameter Fine-Tuning (FPT)

#### QFT: Quantized Full-parameter Tuning of LLMs with Affordable Resources

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

#### Full Parameter Fine-tuning for Large Language Models with Limited Resources (LOMO)

paper link: [here](https://arxiv.org/pdf/2306.09782)

citation: 
```bibtex
@article{lv2023full,
  title={Full Parameter Fine-tuning for Large Language Models with Limited Resources},
  author={Lv, Kai and Yang, Yuqing and Liu, Tengxiao and Gao, Qinghui and Guo, Qipeng and Qiu, Xipeng},
  journal={arXiv preprint arXiv:2306.09782},
  year={2023}
}
```


### Parameter-Efficient Fine-Tuning (PEFT)

#### RoSA: Accurate Parameter-Efficient Fine-Tuning via Robust Adaptation

paper link: [here](https://arxiv.org/pdf/2401.04679.pdf)

citation:

```bibtex
@misc{nikdan2024rosa,
      title={RoSA: Accurate Parameter-Efficient Fine-Tuning via Robust Adaptation}, 
      author={Mahdi Nikdan and Soroush Tabesh and Dan Alistarh},
      year={2024},
      eprint={2401.04679},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models

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

#### LongLoRA: Efficient fine-tuning of long-context large language models [`AREAD`]

paper link: [here](https://arxiv.org/pdf/2309.12307.pdf)

citation: 
```bibtex
@article{chen2023longlora,
  title={Longlora: Efficient fine-tuning of long-context large language models},
  author={Chen, Yukang and Qian, Shengju and Tang, Haotian and Lai, Xin and Liu, Zhijian and Han, Song and Jia, Jiaya},
  journal={arXiv preprint arXiv:2309.12307},
  year={2023}
}
```

#### LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition

paper link: [here](https://arxiv.org/pdf/2307.13269)
github link: [here](https://github.com/sail-sg/lorahub)
hfhub link: [here](https://huggingface.co/lorahub)

citation:

```bibtex
@misc{huang2024lorahub,
      title={LoraHub: Efficient Cross-Task Generalization via Dynamic LoRA Composition}, 
      author={Chengsong Huang and Qian Liu and Bill Yuchen Lin and Tianyu Pang and Chao Du and Min Lin},
      year={2024},
      eprint={2307.13269},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### On the effectiveness of parameter-efficient fine-tuning

paper link: [here](https://ojs.aaai.org/index.php/AAAI/article/download/26505/26277)

citation: 
```bibtex
@inproceedings{fu2023effectiveness,
  title={On the effectiveness of parameter-efficient fine-tuning},
  author={Fu, Zihao and Yang, Haoran and So, Anthony Man-Cho and Lam, Wai and Bing, Lidong and Collier, Nigel},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
  volume={37},
  number={11},
  pages={12799--12807},
  year={2023}
}
```
    

#### Qlora: Efficient finetuning of quantized llms

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

#### Hypertuning: Toward adapting large language models without back-propagation

paper link: [here](https://proceedings.mlr.press/v202/phang23a.html)

citation:

```bibtex
@inproceedings{phang2023hypertuning,
  title={Hypertuning: Toward adapting large language models without back-propagation},
  author={Phang, Jason and Mao, Yi and He, Pengcheng and Chen, Weizhu},
  booktitle={International Conference on Machine Learning},
  pages={27854--27875},
  year={2023},
  organization={PMLR}
}
```


#### LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models

paper link: [here](https://arxiv.org/pdf/2304.01933.pdf)

citation:

```bibtex
@misc{hu2023llmadapters,
      title={LLM-Adapters: An Adapter Family for Parameter-Efficient Fine-Tuning of Large Language Models}, 
      author={Zhiqiang Hu and Lei Wang and Yihuai Lan and Wanyu Xu and Ee-Peng Lim and Lidong Bing and Xing Xu and Soujanya Poria and Roy Ka-Wei Lee},
      year={2023},
      eprint={2304.01933},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
    

#### Scaling down to scale up: A guide to parameter-efficient fine-tuning

paper link: [here](https://arxiv.org/pdf/2303.15647)

citation: 
```bibtex
@article{lialin2023scaling,
  title={Scaling down to scale up: A guide to parameter-efficient fine-tuning},
  author={Lialin, Vladislav and Deshpande, Vijeta and Rumshisky, Anna},
  journal={arXiv preprint arXiv:2303.15647},
  year={2023}
}
```


#### Scaling instruction-finetuned language models

paper link: [here](https://arxiv.org/pdf/2210.11416.pdf)

citation: 
```bibtex
@article{chung2022scaling,
  title={Scaling instruction-finetuned language models},
  author={Chung, Hyung Won and Hou, Le and Longpre, Shayne and Zoph, Barret and Tay, Yi and Fedus, William and Li, Yunxuan and Wang, Xuezhi and Dehghani, Mostafa and Brahma, Siddhartha and others},
  journal={arXiv preprint arXiv:2210.11416},
  year={2022}
}
```

#### PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods

blog link: [here](https://huggingface.co/blog/peft)

github link: [here](https://github.com/huggingface/peft)
    
citation:

```bibtex
@Misc{peft,
  title = {PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods},
  author = {Sourab Mangrulkar and Sylvain Gugger and Lysandre Debut and Younes Belkada and Sayak Paul and Benjamin Bossan},
  howpublished = {\url{https://github.com/huggingface/peft}},
  year = {2022}
}
```

#### Delta tuning: A comprehensive study of parameter efficient methods for pre-trained language models

paper link: [here](https://arxiv.org/pdf/2203.06904)

citation: 
```bibtex
@article{ding2022delta,
  title={Delta tuning: A comprehensive study of parameter efficient methods for pre-trained language models},
  author={Ding, Ning and Qin, Yujia and Yang, Guang and Wei, Fuchao and Yang, Zonghan and Su, Yusheng and Hu, Shengding and Chen, Yulin and Chan, Chi-Min and Chen, Weize and others},
  journal={arXiv preprint arXiv:2203.06904},
  year={2022}
}
```

#### The power of scale for parameter-efficient prompt tuning

paper link: [here](https://arxiv.org/pdf/2104.08691)

citation: 
```bibtex
@article{lester2021power,
  title={The power of scale for parameter-efficient prompt tuning},
  author={Lester, Brian and Al-Rfou, Rami and Constant, Noah},
  journal={arXiv preprint arXiv:2104.08691},
  year={2021}
}
```
    


#### Prefix-tuning: Optimizing continuous prompts for generation

paper link: [here](https://arxiv.org/pdf/2101.00190)

citation: 
```bibtex
@article{li2021prefix,
  title={Prefix-tuning: Optimizing continuous prompts for generation},
  author={Li, Xiang Lisa and Liang, Percy},
  journal={arXiv preprint arXiv:2101.00190},
  year={2021}
}
```
    
#### Lora: Low-rank adaptation of large language models

paper link: [here](https://arxiv.org/pdf/2106.09685.pdf%C2%A0)

citation: 
```bibtex
@article{hu2021lora,
  title={Lora: Low-rank adaptation of large language models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}
```


### Memory-Efficient Fine-Tuning (MEFT)


#### Make Your Pre-trained Model Reversible: From Parameter to Memory Efficient Fine-Tuning

paper link: [here](https://arxiv.org/pdf/2306.00477)

citation: 
```bibtex
@article{liao2023make,
  title={Make Your Pre-trained Model Reversible: From Parameter to Memory Efficient Fine-Tuning},
  author={Liao, Baohao and Tan, Shaomu and Monz, Christof},
  journal={arXiv preprint arXiv:2306.00477},
  year={2023}
}
```


    
    
    