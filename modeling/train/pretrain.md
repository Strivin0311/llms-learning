# Pretraining for LLMs
*Here're some resources about pretraining methodologies for LLMs*


## Efficient Pretraining

#### Efficient Online Data Mixing For Language Model Pre-Training [`READ`]

paper link: [here](https://arxiv.org/pdf/2312.02406.pdf)

citation:
```bibtex
@misc{albalak2023efficient,
      title={Efficient Online Data Mixing For Language Model Pre-Training}, 
      author={Alon Albalak and Liangming Pan and Colin Raffel and William Yang Wang},
      year={2023},
      eprint={2312.02406},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### Train Faster, Perform Better: Modular Adaptive Training in Over-Parameterized Models [`UNREAD`]

paper link: [here](https://openreview.net/pdf?id=dWDEBW2raJ)

citation: 
```bibtex
@inproceedings{shi2023train,
  title={Train Faster, Perform Better: Modular Adaptive Training in Over-Parameterized Models},
  author={Shi, Yubin and Chen, Yixuan and Dong, Mingzhi and Yang, Xiaochen and Li, Dongsheng and Wang, Yujiang and Dick, Robert P and Lv, Qin and Zhao, Yingying and Yang, Fan and others},
  booktitle={Thirty-seventh Conference on Neural Information Processing Systems},
  year={2023}
}
```


#### Transcending Scaling Laws with 0.1% Extra Compute (UL2R, U-PaLM) [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2210.11399.pdf)

citation:
```bibtex
@misc{tay2022transcending,
      title={Transcending Scaling Laws with 0.1% Extra Compute}, 
      author={Yi Tay and Jason Wei and Hyung Won Chung and Vinh Q. Tran and David R. So and Siamak Shakeri and Xavier Garcia and Huaixiu Steven Zheng and Jinfeng Rao and Aakanksha Chowdhery and Denny Zhou and Donald Metzler and Slav Petrov and Neil Houlsby and Quoc V. Le and Mostafa Dehghani},
      year={2022},
      eprint={2210.11399},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
    

#### 1-bit Adam: Communication Efficient Large-Scale Training with Adam's Convergence Speed [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2102.02888.pdf)

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


#### The reversible residual network: Backpropagation without storing activations (RevNet) [`UNREAD`]

paper link: [here](https://proceedings.neurips.cc/paper_files/paper/2017/file/f9be311e65d81a9ad8150a60844bb94c-Paper.pdf)

citation: 
```bibtex
@article{gomez2017reversible,
  title={The reversible residual network: Backpropagation without storing activations},
  author={Gomez, Aidan N and Ren, Mengye and Urtasun, Raquel and Grosse, Roger B},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}
```


#### Training deep nets with sublinear memory cost [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/1604.06174)

citation: 
```bibtex
@article{chen2016training,
  title={Training deep nets with sublinear memory cost},
  author={Chen, Tianqi and Xu, Bing and Zhang, Chiyuan and Guestrin, Carlos},
  journal={arXiv preprint arXiv:1604.06174},
  year={2016}
}
```


## Effective Pretraining

#### Skill-it! A data-driven skills framework for understanding and training language models [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2307.14430)

citation: 
```bibtex
@article{chen2023skill,
  title={Skill-it! A data-driven skills framework for understanding and training language models},
  author={Chen, Mayee F and Roberts, Nicholas and Bhatia, Kush and Wang, Jue and Zhang, Ce and Sala, Frederic and R{\'e}, Christopher},
  journal={arXiv preprint arXiv:2307.14430},
  year={2023}
}
```

#### UL2: Unifying Language Learning Paradigms (MoD) [`UNREAD`]

paper link: [here](https://arxiv.org/pdf/2205.05131.pdf)

citation:
```bibtex
@misc{tay2023ul2,
      title={UL2: Unifying Language Learning Paradigms}, 
      author={Yi Tay and Mostafa Dehghani and Vinh Q. Tran and Xavier Garcia and Jason Wei and Xuezhi Wang and Hyung Won Chung and Siamak Shakeri and Dara Bahri and Tal Schuster and Huaixiu Steven Zheng and Denny Zhou and Neil Houlsby and Donald Metzler},
      year={2023},
      eprint={2205.05131},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### Training compute-optimal large language models [`READ`]

paper link: [here](https://arxiv.org/pdf/2203.15556)

citation: 
```bibtex
@article{hoffmann2022training,
  title={Training compute-optimal large language models},
  author={Hoffmann, Jordan and Borgeaud, Sebastian and Mensch, Arthur and Buchatskaya, Elena and Cai, Trevor and Rutherford, Eliza and Casas, Diego de Las and Hendricks, Lisa Anne and Welbl, Johannes and Clark, Aidan and others},
  journal={arXiv preprint arXiv:2203.15556},
  year={2022}
}
```
    

#### Selfdoc: Self-supervised document representation learning [`READ`]

paper link: [here](https://openaccess.thecvf.com/content/CVPR2021/papers/Li_SelfDoc_Self-Supervised_Document_Representation_Learning_CVPR_2021_paper.pdf)

citation: 
```bibtex
@inproceedings{li2021selfdoc,
  title={Selfdoc: Self-supervised document representation learning},
  author={Li, Peizhao and Gu, Jiuxiang and Kuen, Jason and Morariu, Vlad I and Zhao, Handong and Jain, Rajiv and Manjunatha, Varun and Liu, Hongfu},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={5652--5660},
  year={2021}
}
```


## Pretraining Corpus


### General


#### RedPajama: an Open Dataset for Training Large Language Models [`READ`]

blog link: [here](https://together.ai/blog/redpajama-data-v2)

github link: [here](https://github.com/togethercomputer/RedPajama-Data)

dataset link: [here](https://huggingface.co/datasets/togethercomputer/RedPajama-Data-V2)

citation: 
```bibtex
@software{together2023redpajama,
  author = {Together Computer},
  title = {RedPajama: an Open Dataset for Training Large Language Models},
  month = October,
  year = 2023,
  url = {https://github.com/togethercomputer/RedPajama-Data}
}
```


#### The BigScience ROOTS Corpus: A 1.6TB Composite Multilingual Dataset [`UNREAD`]

paper link: [here](https://proceedings.neurips.cc/paper_files/paper/2022/file/ce9e92e3de2372a4b93353eb7f3dc0bd-Paper-Datasets_and_Benchmarks.pdf)

github link: [here](https://github.com/bigscience-workshop/data-preparation)

dataset link: [here](https://huggingface.co/bigscience-data)

citation: 
```bibtex
@article{laurenccon2022bigscience,
  title={The bigscience roots corpus: A 1.6 tb composite multilingual dataset},
  author={Lauren{\c{c}}on, Hugo and Saulnier, Lucile and Wang, Thomas and Akiki, Christopher and Villanova del Moral, Albert and Le Scao, Teven and Von Werra, Leandro and Mou, Chenghao and Gonz{\'a}lez Ponferrada, Eduardo and Nguyen, Huu and others},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  pages={31809--31826},
  year={2022}
}
```


#### The Pile: An 800GB Dataset of Diverse Text for Language Modeling [`READ`]

paper link: [here](https://arxiv.org/pdf/2101.00027.pdf)

github link: [here](https://github.com/EleutherAI/the-pile)

dataset link: [here](https://the-eye.eu/public/AI/pile/)

citation: 
```bibtex
@misc{gao2020pile,
      title={The Pile: An 800GB Dataset of Diverse Text for Language Modeling}, 
      author={Leo Gao and Stella Biderman and Sid Black and Laurence Golding and Travis Hoppe and Charles Foster and Jason Phang and Horace He and Anish Thite and Noa Nabeshima and Shawn Presser and Connor Leahy},
      year={2020},
      eprint={2101.00027},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



### Math

#### Generative AI for Math: Part I -- MathPile: A Billion-Token-Scale Pretraining Corpus for Math [`READ`]

paper link: [here](https://arxiv.org/pdf/2312.17120)

github link: [here](https://github.com/GAIR-NLP/MathPile/)

dataset link: [here](https://huggingface.co/datasets/GAIR/MathPile)

citation: 
```bibtex
@misc{wang2023generative,
      title={Generative AI for Math: Part I -- MathPile: A Billion-Token-Scale Pretraining Corpus for Math}, 
      author={Zengzhi Wang and Rui Xia and Pengfei Liu},
      year={2023},
      eprint={2312.17120},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### Llemma: An Open Language Model For Mathematics (Proof-Pile-2) [`READ`]

paper link: [here](https://arxiv.org/pdf/2310.10631.pdf)

github link: [here](https://github.com/EleutherAI/math-lm)

dataset link: [here](https://huggingface.co/datasets/EleutherAI/proof-pile-2)

citation:
```bibtex
@misc{azerbayev2023llemma,
      title={Llemma: An Open Language Model For Mathematics}, 
      author={Zhangir Azerbayev and Hailey Schoelkopf and Keiran Paster and Marco Dos Santos and Stephen McAleer and Albert Q. Jiang and Jia Deng and Stella Biderman and Sean Welleck},
      year={2023},
      eprint={2310.10631},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```


#### OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text [`READ`]

paper link: [here](https://arxiv.org/pdf/2310.06786)

github link: [here](https://github.com/keirp/OpenWebMath)

dataset link: [here](https://huggingface.co/datasets/open-web-math/open-web-math)

citation:
```bibtex
@misc{paster2023openwebmath,
      title={OpenWebMath: An Open Dataset of High-Quality Mathematical Web Text}, 
      author={Keiran Paster and Marco Dos Santos and Zhangir Azerbayev and Jimmy Ba},
      year={2023},
      eprint={2310.06786},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```


#### MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models (MetaMathQA) [`READ`]

paper link: [here](https://arxiv.org/pdf/2309.12284)

github link: [here](https://github.com/meta-math/MetaMath)

dataset link: [here](https://huggingface.co/datasets/meta-math/MetaMathQA)

citation:
```bibtex
@misc{yu2023metamath,
      title={MetaMath: Bootstrap Your Own Mathematical Questions for Large Language Models}, 
      author={Longhui Yu and Weisen Jiang and Han Shi and Jincheng Yu and Zhengying Liu and Yu Zhang and James T. Kwok and Zhenguo Li and Adrian Weller and Weiyang Liu},
      year={2023},
      eprint={2309.12284},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```



#### Let's Verify Step by Step (PRM800K) [`READ`]

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