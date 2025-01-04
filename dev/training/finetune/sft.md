# Supervised Fine-Tuning on LLMs
*Here're some resources about Supervised Finetuning strategies on LLMs, especially parameter-efficient fine-tuning (PEFT)and memory-efficient fine-tuning (MEFT)*


## Method


#### Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients

tag: `Q-GaLore` | `MEFT` | `Meta`

paper link: [here](https://arxiv.org/pdf/2407.08296)

code link: [here](https://github.com/VITA-Group/Q-GaLore)

citation:

```bibtex
@misc{zhang2024qgalorequantizedgaloreint4,
      title={Q-GaLore: Quantized GaLore with INT4 Projection and Layer-Adaptive Low-Rank Gradients}, 
      author={Zhenyu Zhang and Ajay Jaiswal and Lu Yin and Shiwei Liu and Jiawei Zhao and Yuandong Tian and Zhangyang Wang},
      year={2024},
      eprint={2407.08296},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2407.08296}, 
}
```


#### LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning

tag: `LISA` | `PEFT` | `NIPS24` | `HKU`

paper link: [here](https://openreview.net/pdf?id=L8ifDX5XNq)

code link: [here](https://github.com/OptimalScale/LMFlow)

citation:

```bibtex
@misc{pan2024lisalayerwiseimportancesampling,
      title={LISA: Layerwise Importance Sampling for Memory-Efficient Large Language Model Fine-Tuning}, 
      author={Rui Pan and Xiang Liu and Shizhe Diao and Renjie Pi and Jipeng Zhang and Chi Han and Tong Zhang},
      year={2024},
      eprint={2403.17919},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2403.17919}, 
}
```


#### GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection

tag: `GaLore` | `MEFT` | `ICML24` | `Meta`

paper link: [here](https://arxiv.org/pdf/2403.03507)

code link: [here](https://github.com/jiaweizzhao/GaLore)

follow-up work: [here](https://arxiv.org/pdf/2407.08296)

citation:

```bibtex
@misc{zhao2024galorememoryefficientllmtraining,
      title={GaLore: Memory-Efficient LLM Training by Gradient Low-Rank Projection}, 
      author={Jiawei Zhao and Zhenyu Zhang and Beidi Chen and Zhangyang Wang and Anima Anandkumar and Yuandong Tian},
      year={2024},
      eprint={2403.03507},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2403.03507}, 
}
```


#### RoSA: Accurate Parameter-Efficient Fine-Tuning via Robust Adaptation

tag: `RoSA` | `PEFT` | `ICML24`

paper link: [here](https://arxiv.org/pdf/2401.04679.pdf)

code link: [here](https://github.com/IST-DASLab/RoSA)

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


#### A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA

tag: `Rank Stabilization` | `LoRA` | `Tenyx`

Paper link: [here](https://arxiv.org/pdf/2312.03732)

citation:

```bibtex
@misc{kalajdzievski2023rankstabilizationscalingfactor,
      title={A Rank Stabilization Scaling Factor for Fine-Tuning with LoRA}, 
      author={Damjan Kalajdzievski},
      year={2023},
      eprint={2312.03732},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2312.03732}, 
}
```


#### QFT: Quantized Full-parameter Tuning of LLMs with Affordable Resources

tag: `QFT` | `MEFT` | `UCAS` | `UC Berkeley`

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

tag: `QA-LoRA` | `PEFT` | `ICLR24` | `Huawei`

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


#### LongLoRA: Efficient fine-tuning of long-context large language models

tag: `LongLoRA` | `PEFT` | `ICLR24` | `CUHK` | `MIT`

paper link: [here](https://arxiv.org/pdf/2309.12307.pdf)

code link: [here](https://github.com/dvlab-research/LongLoRA)

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

tag: `LoraHub` | `PEFT` | `COLM24` | `Sea AI Lab` | `Allen AI`

paper link: [here](https://arxiv.org/pdf/2307.13269)

code link: [here](https://github.com/sail-sg/lorahub)

modelhub link: [here](https://huggingface.co/lorahub)

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


#### Full Parameter Fine-tuning for Large Language Models with Limited Resources

tag: `LOMO` | `MEFT` | `ACL24` | `Shanghai AI Laboratory` | `Fudan University`

paper link: [here](https://aclanthology.org/2024.acl-long.445.pdf)

code link: [here](https://github.com/OpenLMLab/LOMO)

citation:

```bibtex
@article{lv2023full,
  title={Full Parameter Fine-tuning for Large Language Models with Limited Resources},
  author={Lv, Kai and Yang, Yuqing and Liu, Tengxiao and Gao, Qinghui and Guo, Qipeng and Qiu, Xipeng},
  journal={arXiv preprint arXiv:2306.09782},
  year={2023}
}
```


#### Make Your Pre-trained Model Reversible: From Parameter to Memory Efficient Fine-Tuning

tag: `MEFT` | `Rev-ViT` | `NIPS23`

paper link: [here](https://arxiv.org/pdf/2306.00477)

code link: [here](https://github.com/BaohaoLiao/mefts)

citation:

```bibtex
@article{liao2023make,
  title={Make Your Pre-trained Model Reversible: From Parameter to Memory Efficient Fine-Tuning},
  author={Liao, Baohao and Tan, Shaomu and Monz, Christof},
  journal={arXiv preprint arXiv:2306.00477},
  year={2023}
}
```
    

#### Qlora: Efficient finetuning of quantized llms

tag: `QLoRA` | `PEFT` | `NIPS23` | `University of Washington`

paper link: [here](https://arxiv.org/pdf/2305.14314)

code link: [here](https://github.com/artidoro/qlora)

tutorial link: [here](../../tutorial/notebook/tutorial_qlora.ipynb)

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

tag: `Hyper-Tuning` | `Prefix Tuning` | `PEFT` | `ICML23` | `Microsoft`

paper link: [here](https://proceedings.mlr.press/v202/phang23a/phang23a.pdf)

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

tag: `LLM-Adapters` | `PEFT` | `EMNLP23`

paper link: [here](https://aclanthology.org/2023.emnlp-main.319.pdf)

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


#### PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods

tag: `PEFT` | `HuggingFace`

blog link: [here](https://huggingface.co/blog/peft)

code link: [here](https://github.com/huggingface/peft)
    
citation:

```bibtex
@Misc{peft,
  title = {PEFT: State-of-the-art Parameter-Efficient Fine-Tuning methods},
  author = {Sourab Mangrulkar and Sylvain Gugger and Lysandre Debut and Younes Belkada and Sayak Paul and Benjamin Bossan},
  howpublished = {\url{https://github.com/huggingface/peft}},
  year = {2022}
}
```


#### The power of scale for parameter-efficient prompt tuning

tag: `Prpompt Tuning` | `PEFT` | `EMNLP21` | `Google`

paper link: [here](https://aclanthology.org/2021.emnlp-main.243.pdf)

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

tag: `Prefix Tuning` | `PEFT` | `ACL21` | `Stanford University`

paper link: [here](https://aclanthology.org/2021.acl-long.353.pdf)

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

tag: `LoRA` | `PEFT` | `ICLR22` | `Microsoft`

paper link: [here](https://openreview.net/pdf?id=nZeVKeeFYf9)

code link: [here](https://github.com/microsoft/LoRA)

citation:

```bibtex
@article{hu2021lora,
  title={Lora: Low-rank adaptation of large language models},
  author={Hu, Edward J and Shen, Yelong and Wallis, Phillip and Allen-Zhu, Zeyuan and Li, Yuanzhi and Wang, Shean and Wang, Lu and Chen, Weizhu},
  journal={arXiv preprint arXiv:2106.09685},
  year={2021}
}
```


## Survey


#### Scaling down to scale up: A guide to parameter-efficient fine-tuning

tag: `PEFT Survey` | `UML`

paper link: [here](https://arxiv.org/pdf/2303.15647)

code link: [here](https://github.com/guitaricet/peft_comparison)

citation:

```bibtex
@article{lialin2023scaling,
  title={Scaling down to scale up: A guide to parameter-efficient fine-tuning},
  author={Lialin, Vladislav and Deshpande, Vijeta and Rumshisky, Anna},
  journal={arXiv preprint arXiv:2303.15647},
  year={2023}
}
```


#### On the effectiveness of parameter-efficient fine-tuning

tag: `PEFT Sruvey` | `AAAI23` | `DAMO Academy` | `Alibaba Group`

paper link: [here](https://arxiv.org/pdf/2211.15583)

code link: [here](https://github.com/fuzihaofzh/AnalyzeParameterEfficientFinetune)

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


#### Delta tuning: A comprehensive study of parameter efficient methods for pre-trained language models

tag: `Delta Tuning` | `PEFT Survey` | `Tsinghua University`

paper link: [here](https://arxiv.org/pdf/2203.06904)

code link: [here](https://github.com/thunlp/OpenDelta)

citation:

```bibtex
@article{ding2022delta,
  title={Delta tuning: A comprehensive study of parameter efficient methods for pre-trained language models},
  author={Ding, Ning and Qin, Yujia and Yang, Guang and Wei, Fuchao and Yang, Zonghan and Su, Yusheng and Hu, Shengding and Chen, Yulin and Chan, Chi-Min and Chen, Weize and others},
  journal={arXiv preprint arXiv:2203.06904},
  year={2022}
}
```






    
    
    