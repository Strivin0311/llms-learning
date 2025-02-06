# Prompt Learning and Engineering on LLMs
*Here're some resources about prompt learning and engineering on LLMs, especially in-context learning (ICL) and chain-of-thoughts (CoT) reasoning*


## Method


#### Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs

tag: `MIPRO` | `DSPy` | `EMNLP24` | `Stanford University`

paper link: [here](https://aclanthology.org/2024.emnlp-main.525.pdf)

code link: [here](https://github.com/stanfordnlp/dspy)

citation:

```bibtex
@misc{opsahlong2024optimizinginstructionsdemonstrationsmultistage,
      title={Optimizing Instructions and Demonstrations for Multi-Stage Language Model Programs}, 
      author={Krista Opsahl-Ong and Michael J Ryan and Josh Purtell and David Broman and Christopher Potts and Matei Zaharia and Omar Khattab},
      year={2024},
      eprint={2406.11695},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.11695}, 
}
```


#### Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization

tag: `GrokkedTransformer` | `Grokking` | `NIPS24` | `CMU`

paper link: [here](https://arxiv.org/pdf/2405.15071)

code link: [here](https://github.com/OSU-NLP-Group/GrokkedTransformer)

citation:

```bibtex
@misc{wang2024grokkedtransformersimplicitreasoners,
      title={Grokked Transformers are Implicit Reasoners: A Mechanistic Journey to the Edge of Generalization}, 
      author={Boshi Wang and Xiang Yue and Yu Su and Huan Sun},
      year={2024},
      eprint={2405.15071},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2405.15071}, 
}
```


#### PathFinder: Guided Search over Multi-Step Reasoning Paths

tag: `PathFinder` | `NIPS23 Ro-FoMo Workshop` | `Meta`

paper link: [here](https://arxiv.org/pdf/2312.05180.pdf)

citation:

```bibtex
@misc{golovneva2023pathfinder,
      title={PathFinder: Guided Search over Multi-Step Reasoning Paths}, 
      author={Olga Golovneva and Sean O'Brien and Ramakanth Pasunuru and Tianlu Wang and Luke Zettlemoyer and Maryam Fazel-Zarandi and Asli Celikyilmaz},
      year={2023},
      eprint={2312.05180},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### Plum: Prompt Learning using Metaheuristic

tag: `Plum` | `ACL24` | `HKU`

paper link: [here](https://aclanthology.org/2024.findings-acl.129.pdf)

code link: [here](https://github.com/research4pan/Plum)

citation:

```bibtex
@article{pan2023plum,
  title={Plum: Prompt Learning using Metaheuristic},
  author={Pan, Rui and Xing, Shuo and Diao, Shizhe and Liu, Xiang and Shum, Kashun and Zhang, Jipeng and Zhang, Tong},
  journal={arXiv preprint arXiv:2311.08364},
  year={2023}
}
```


#### Ddcot: Duty-distinct chain-of-thought prompting for multimodal reasoning in language models

tag: `DDCoT` | `CoT` | `Multi-Modal` | `NIPS23` | `ShanghaiTech University`

paper link: [here](https://arxiv.org/pdf/2310.16436)

code link: [here](https://github.com/SooLab/DDCOT)

homepage link: [here](https://toneyaya.github.io/ddcot/)

citation:

```bibtex
@article{zheng2023ddcot,
  title={Ddcot: Duty-distinct chain-of-thought prompting for multimodal reasoning in language models},
  author={Zheng, Ge and Yang, Bin and Tang, Jiajin and Zhou, Hong-Yu and Yang, Sibei},
  journal={arXiv preprint arXiv:2310.16436},
  year={2023}
}
```


#### DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines

tag: `DSPy` | `ICLR24` | `Stanford University` | `UCB` | `CMU`

paper link: [here](https://openreview.net/pdf?id=sY5N0zY5Od)

code link: [here](https://github.com/stanfordnlp/dspy)

follow-up work: [here](https://arxiv.org/pdf/2406.11695)

citation:

```bibtex
@misc{khattab2023dspycompilingdeclarativelanguage,
      title={DSPy: Compiling Declarative Language Model Calls into Self-Improving Pipelines}, 
      author={Omar Khattab and Arnav Singhvi and Paridhi Maheshwari and Zhiyuan Zhang and Keshav Santhanam and Sri Vardhamanan and Saiful Haq and Ashutosh Sharma and Thomas T. Joshi and Hanna Moazam and Heather Miller and Matei Zaharia and Christopher Potts},
      year={2023},
      eprint={2310.03714},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2310.03714}, 
}
```


#### Evoke: Evoking Critical Thinking Abilities in LLMs via Reviewer-Author Prompt Editing

tag: `Evoke` | `ICLR24` | `Microsoft` | `University of Washington`

paper link: [here](https://openreview.net/pdf?id=OXv0zQ1umU)

citation:

```bibtex
@misc{hu2023evoke,
      title={Evoke: Evoking Critical Thinking Abilities in LLMs via Reviewer-Author Prompt Editing}, 
      author={Xinyu Hu and Pengfei Tang and Simiao Zuo and Zihan Wang and Bowen Song and Qiang Lou and Jian Jiao and Denis Charles},
      year={2023},
      eprint={2310.13855},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### Graph of thoughts: Solving elaborate problems with large language models

tag: `GoT` | `AAAI24`

paper link: [here](https://arxiv.org/pdf/2308.09687.pdf)

code link: [here](https://github.com/spcl/graph-of-thoughts)

citation:

```bibtex
@article{besta2023graph,
  title={Graph of thoughts: Solving elaborate problems with large language models},
  author={Besta, Maciej and Blach, Nils and Kubicek, Ales and Gerstenberger, Robert and Gianinazzi, Lukas and Gajda, Joanna and Lehmann, Tomasz and Podstawski, Michal and Niewiadomski, Hubert and Nyczyk, Piotr and others},
  journal={arXiv preprint arXiv:2308.09687},
  year={2023}
}
```


#### Joint Prompt Optimization of Stacked LLMs using Variational Inference

tag: `DLN` | `NIPS23` | `Microsoft`

paper link: [here](https://arxiv.org/pdf/2306.12509.pdf)

code link: [here](https://github.com/microsoft/deep-language-networks)

```bibtex
@inproceedings{sordoni2023joint,
  title={Joint Prompt Optimization of Stacked LLMs using Variational Inference},
  author={Sordoni, Alessandro and Yuan, Xingdi and C{\^o}t{\'e}, Marc-Alexandre and Pereira, Matheus and Trischler, Adam and Xiao, Ziang and Hosseini, Arian and Niedtner, Friederike and Le Roux, Nicolas},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```


#### Tree of thoughts: Deliberate problem solving with large language models

tag: `ToT` | `NIPS23` | `Google DeepMind` | `Princeton University`

paper link: [here](https://arxiv.org/pdf/2305.10601)

code link: [here](https://github.com/princeton-nlp/tree-of-thought-llm)

citation:

```bibtex
@article{yao2023tree,
  title={Tree of thoughts: Deliberate problem solving with large language models},
  author={Yao, Shunyu and Yu, Dian and Zhao, Jeffrey and Shafran, Izhak and Griffiths, Thomas L and Cao, Yuan and Narasimhan, Karthik},
  journal={arXiv preprint arXiv:2305.10601},
  year={2023}
}
```


#### Query2doc: Query Expansion with Large Language Models

tag: `Query2Doc` | `EMNLP23` | `Microsoft`

paper link: [here](https://aclanthology.org/2023.emnlp-main.585.pdf)

citation:

```bibtex
@misc{jagerman2023query,
      title={Query Expansion by Prompting Large Language Models}, 
      author={Rolf Jagerman and Honglei Zhuang and Zhen Qin and Xuanhui Wang and Michael Bendersky},
      year={2023},
      eprint={2305.03653},
      archivePrefix={arXiv},
      primaryClass={cs.IR}
}
```


#### Chameleon: Plug-and-play compositional reasoning with large language models

tag: `Chameleon` | `NIPS23` | `Microsoft` | `UCLA`

paper link: [here](https://arxiv.org/pdf/2304.09842)

code link: [here](https://github.com/lupantech/chameleon-llm)

homepage link: [here](https://chameleon-llm.github.io/)

citation:

```bibtex
@article{lu2023chameleon,
  title={Chameleon: Plug-and-play compositional reasoning with large language models},
  author={Lu, Pan and Peng, Baolin and Cheng, Hao and Galley, Michel and Chang, Kai-Wei and Wu, Ying Nian and Zhu, Song-Chun and Gao, Jianfeng},
  journal={arXiv preprint arXiv:2304.09842},
  year={2023}
}
```


#### Pal: Program-aided language models

tag: `PaL` | `ICML23` | `CMU`

paper link: [here](https://proceedings.mlr.press/v202/gao23f/gao23f.pdf)

code link: [here](https://github.com/reasoning-machines/pal)

homepage link: [here](https://reasonwithpal.com/)

citation:

```bibtex
@inproceedings{gao2023pal,
  title={Pal: Program-aided language models},
  author={Gao, Luyu and Madaan, Aman and Zhou, Shuyan and Alon, Uri and Liu, Pengfei and Yang, Yiming and Callan, Jamie and Neubig, Graham},
  booktitle={International Conference on Machine Learning},
  pages={10764--10799},
  year={2023},
  organization={PMLR}
}
```

#### Visual programming: Compositional visual reasoning without training

tag: `VisProg` | `CVPR23` | `Allen AI`

paper link: [here](https://openaccess.thecvf.com/content/CVPR2023/papers/Gupta_Visual_Programming_Compositional_Visual_Reasoning_Without_Training_CVPR_2023_paper.pdf)

code link: [here](https://github.com/allenai/visprog)

homepage link: [here](https://prior.allenai.org/projects/visprog)

citation:

```bibtex
@inproceedings{gupta2023visual,
  title={Visual programming: Compositional visual reasoning without training},
  author={Gupta, Tanmay and Kembhavi, Aniruddha},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={14953--14962},
  year={2023}
}
```


#### Maple: Multi-modal prompt learning

tag: `MaPLe` | `Multi-Modal` | `CVPR23`

paper link: [here](https://openaccess.thecvf.com/content/CVPR2023/papers/Khattak_MaPLe_Multi-Modal_Prompt_Learning_CVPR_2023_paper.pdf)

code link: [here](https://github.com/muzairkhattak/multimodal-prompt-learning)

citation:

```bibtex
@inproceedings{khattak2023maple,
  title={Maple: Multi-modal prompt learning},
  author={Khattak, Muhammad Uzair and Rasheed, Hanoona and Maaz, Muhammad and Khan, Salman and Khan, Fahad Shahbaz},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19113--19122},
  year={2023}
}
```


#### Structured prompting: Scaling in-context learning to 1,000 examples

tag: `Structured Prompting` | `Microsoft`

paper link: [here](https://arxiv.org/pdf/2212.06713)

code link: [here](https://github.com/microsoft/LMOps)

citation:

```bibtex
@article{hao2022structured,
  title={Structured prompting: Scaling in-context learning to 1,000 examples},
  author={Hao, Yaru and Sun, Yutao and Dong, Li and Han, Zhixiong and Gu, Yuxian and Wei, Furu},
  journal={arXiv preprint arXiv:2212.06713},
  year={2022}
}
```


#### Large language models can self-improve

tag: `Self Improve` | `EMNLP23` | `Google`

paper link: [here](https://aclanthology.org/2023.emnlp-main.67.pdf)

citation:

```bibtex
@article{huang2022large,
  title={Large language models can self-improve},
  author={Huang, Jiaxin and Gu, Shixiang Shane and Hou, Le and Wu, Yuexin and Wang, Xuezhi and Yu, Hongkun and Han, Jiawei},
  journal={arXiv preprint arXiv:2210.11610},
  year={2022}
}
```

#### Large language models are human-level prompt engineers

tag: `APE` | `ICLR23` | `University of Toronto`

paper link: [here](https://arxiv.org/pdf/2211.01910.pdf)

code link: [here](https://github.com/keirp/automatic_prompt_engineer)

citation:

```bibtex
@article{zhou2022large,
  title={Large language models are human-level prompt engineers},
  author={Zhou, Yongchao and Muresanu, Andrei Ioan and Han, Ziwen and Paster, Keiran and Pitis, Silviu and Chan, Harris and Ba, Jimmy},
  journal={arXiv preprint arXiv:2211.01910},
  year={2022}
}
```


#### Conditional prompt learning for vision-language models

tag: `CoCoOp` | `CoOp` | `CVPR22` | `VLMs`

paper link: [here](http://openaccess.thecvf.com/content/CVPR2022/papers/Zhou_Conditional_Prompt_Learning_for_Vision-Language_Models_CVPR_2022_paper.pdf)

code link: [here](https://github.com/KaiyangZhou/CoOp)

citation:

```bibtex
@inproceedings{zhou2022conditional,
  title={Conditional prompt learning for vision-language models},
  author={Zhou, Kaiyang and Yang, Jingkang and Loy, Chen Change and Liu, Ziwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={16816--16825},
  year={2022}
}
```


#### React: Synergizing reasoning and acting in language models

tag: `ReAct` | `ICLR23` | `Google Brain` | `Princeton University`

paper link: [here](https://arxiv.org/pdf/2210.03629.pdf)

code link: [here](https://github.com/ysymyth/ReAct)

homepage link: [here](https://react-lm.github.io/)

citation:

```bibtex
@article{yao2022react,
  title={React: Synergizing reasoning and acting in language models},
  author={Yao, Shunyu and Zhao, Jeffrey and Yu, Dian and Du, Nan and Shafran, Izhak and Narasimhan, Karthik and Cao, Yuan},
  journal={arXiv preprint arXiv:2210.03629},
  year={2022}
}
```


#### Chain-of-thought prompting elicits reasoning in large language models

tag: `CoT` | `NIPS22` | `Google Brain`

paper link: [here](https://proceedings.neurips.cc/paper_files/paper/2022/file/9d5609613524ecf4f15af0f7b31abca4-Paper-Conference.pdf)

citation:

```bibtex
@article{wei2022chain,
  title={Chain-of-thought prompting elicits reasoning in large language models},
  author={Wei, Jason and Wang, Xuezhi and Schuurmans, Dale and Bosma, Maarten and Xia, Fei and Chi, Ed and Le, Quoc V and Zhou, Denny and others},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={24824--24837},
  year={2022}
}
```


#### Language models show human-like content effects on reasoning

tag: `Abstract Reasoning` | `Google DeepMind`

paper link: [here](https://arxiv.org/pdf/2207.07051.pdf)

citation:

```bibtex
@article{dasgupta2022language,
  title={Language models show human-like content effects on reasoning},
  author={Dasgupta, Ishita and Lampinen, Andrew K and Chan, Stephanie CY and Creswell, Antonia and Kumaran, Dharshan and McClelland, James L and Hill, Felix},
  journal={arXiv preprint arXiv:2207.07051},
  year={2022}
}
```


#### RLPrompt: Optimizing Discrete Text Prompts with Reinforcement Learning

tag: `RLPrompt` | `EMNLP22` | `CMU`

paper link: [here](https://aclanthology.org/2022.emnlp-main.222.pdf)

code link: [here](https://github.com/mingkaid/rl-prompt)

citation:

```bibtex
@article{deng2022rlprompt,
  title={Rlprompt: Optimizing discrete text prompts with reinforcement learning},
  author={Deng, Mingkai and Wang, Jianyu and Hsieh, Cheng-Ping and Wang, Yihan and Guo, Han and Shu, Tianmin and Song, Meng and Xing, Eric P and Hu, Zhiting},
  journal={arXiv preprint arXiv:2205.12548},
  year={2022}
}
```


#### Self-consistency improves chain of thought reasoning in language models

tag: `CoT` | `SC` | `Self Consistency` | `ICLR23` | `Google`

paper link: [here](https://arxiv.org/pdf/2203.11171.pdf)

citation:

```bibtex
@article{wang2022self,
  title={Self-consistency improves chain of thought reasoning in language models},
  author={Wang, Xuezhi and Wei, Jason and Schuurmans, Dale and Le, Quoc and Chi, Ed and Narang, Sharan and Chowdhery, Aakanksha and Zhou, Denny},
  journal={arXiv preprint arXiv:2203.11171},
  year={2022}
}
```

#### PromptSource: An Integrated Development Environment and Repository for Natural Language Prompts

tag: `PromptSource` | `ACL22` | `HuggingFace` | `BigScience` | `Brown University`

paper link: [here](https://aclanthology.org/2022.acl-demo.9.pdf)

git link: [here](https://github.com/bigscience-workshop/promptsource)

citation:

```bibtex
@misc{bach2022promptsource,
      title={PromptSource: An Integrated Development Environment and Repository for Natural Language Prompts}, 
      author={Stephen H. Bach and Victor Sanh and Zheng-Xin Yong and Albert Webson and Colin Raffel and Nihal V. Nayak and Abheesht Sharma and Taewoon Kim and M Saiful Bari and Thibault Fevry and Zaid Alyafeai and Manan Dey and Andrea Santilli and Zhiqing Sun and Srulik Ben-David and Canwen Xu and Gunjan Chhablani and Han Wang and Jason Alan Fries and Maged S. Al-shaibani and Shanya Sharma and Urmish Thakker and Khalid Almubarak and Xiangru Tang and Dragomir Radev and Mike Tian-Jian Jiang and Alexander M. Rush},
      year={2022},
      eprint={2202.01279},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```


#### Multitask prompted training enables zero-shot task generalization

tag: `T-Zero` | `Multi-Task` | `ICLR22` | `HuggingFace` | `BigScience` | `Brown University`

paper link: [here](https://arxiv.org/pdf/2110.08207)

code link: [here](https://github.com/bigscience-workshop/t-zero)

citation:

```bibtex
@article{sanh2021multitask,
  title={Multitask prompted training enables zero-shot task generalization},
  author={Sanh, Victor and Webson, Albert and Raffel, Colin and Bach, Stephen H and Sutawika, Lintang and Alyafeai, Zaid and Chaffin, Antoine and Stiegler, Arnaud and Scao, Teven Le and Raja, Arun and others},
  journal={arXiv preprint arXiv:2110.08207},
  year={2021}
}
```

  
## Empirical Study


#### Why think step-by-step? Reasoning emerges from the locality of experience

tag: `Locality of Experience` | `CoT` | `NIPS23` | `Stanford University`

paper link: [here](https://arxiv.org/pdf/2304.03843)

code link: [here](https://github.com/benpry/why-think-step-by-step)

citation:

```bibtex
@article{prystawski2023think,
  title={Why think step-by-step? Reasoning emerges from the locality of experience},
  author={Prystawski, Ben and Goodman, Noah D},
  journal={arXiv preprint arXiv:2304.03843},
  year={2023}
}
```


#### How Do Transformers Learn In-Context Beyond Simple Functions? A Case Study on Learning with Representations

tag: `ICL` | `ICLR24` | `Salesforce AI Research` | `UC Berkeley`

paper link: [here](https://openreview.net/pdf?id=ikwEDva1JZ)

citation:

```bibtex
@article{guo2023transformers,
  title={How Do Transformers Learn In-Context Beyond Simple Functions? A Case Study on Learning with Representations},
  author={Guo, Tianyu and Hu, Wei and Mei, Song and Wang, Huan and Xiong, Caiming and Savarese, Silvio and Bai, Yu},
  journal={arXiv preprint arXiv:2310.10616},
  year={2023}
}
```


#### Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting

tag: `CoT` | `NIPS23` | `NYU`

paper link: [here](https://arxiv.org/pdf/2305.04388.pdf)

code link: [here](https://github.com/milesaturpin/cot-unfaithfulness)

citation:

```bibtex
@article{turpin2023language,
  title={Language Models Don't Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting},
  author={Turpin, Miles and Michael, Julian and Perez, Ethan and Bowman, Samuel R},
  journal={arXiv preprint arXiv:2305.04388},
  year={2023}
}
```

#### Larger language models do in-context learning differently

tag: `SUL-ICL` | `ICL` | `Google Brain` | `Stanford University`

paper link: [here](https://arxiv.org/pdf/2303.03846.pdf)

citation:

```bibtex
@misc{wei2023larger,
      title={Larger language models do in-context learning differently}, 
      author={Jerry Wei and Jason Wei and Yi Tay and Dustin Tran and Albert Webson and Yifeng Lu and Xinyun Chen and Hanxiao Liu and Da Huang and Denny Zhou and Tengyu Ma},
      year={2023},
      eprint={2303.03846},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


## Survey


#### A survey for in-context learning

tag: `ICL Survey` | `EMNLP24` | `ByteDance` | `Shanghai AI Lab` | `Alibaba Group` | `Peking University` | `CMU`

paper link: [here](https://aclanthology.org/2024.emnlp-main.64.pdf)

citation:

```bibtex
@article{dong2022survey,
  title={A survey for in-context learning},
  author={Dong, Qingxiu and Li, Lei and Dai, Damai and Zheng, Ce and Wu, Zhiyong and Chang, Baobao and Sun, Xu and Xu, Jingjing and Sui, Zhifang},
  journal={arXiv preprint arXiv:2301.00234},
  year={2022}
}
```
    

#### Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing

tag: `Prompt Engineering Survey` | `ACM Computing Surveys` | `CMU`

paper link: [here](https://dl.acm.org/doi/pdf/10.1145/3560815?trk=public_post_comment-text)

citation:

```bibtex
@article{liu2023pre,
  title={Pre-train, prompt, and predict: A systematic survey of prompting methods in natural language processing},
  author={Liu, Pengfei and Yuan, Weizhe and Fu, Jinlan and Jiang, Zhengbao and Hayashi, Hiroaki and Neubig, Graham},
  journal={ACM Computing Surveys},
  volume={55},
  number={9},
  pages={1--35},
  year={2023},
  publisher={ACM New York, NY}
}
```