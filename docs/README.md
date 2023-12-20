---
pageClass: home-page
# some data for the components

name: Yibo Liu
profile: /myprofile2.jpeg

socials:
  - title: google-scholar
    icon: "/icons/google-scholar.svg"
    link: https://scholar.google.com/citations?user=FQExy98AAAAJ&hl=en&oi=ao

  - title: github
    icon: "/icons/github.svg"
    link: https://github.com/xDarkLemon

  - title: twitter
    icon: "/icons/twitter.svg"
    link: https://twitter.com/_liuyibo


# cv: https://en.wikipedia.org/wiki/Harry_Potter
bio: Computer Science
email: yl6769 (at) nyu (dot) edu
---

<ProfileSection :frontmatter="$page.frontmatter" />

## About

I am Yibo Liu, a first year Ph.D. student in Computer Science at University of Victoria. Prior to this, I obtained my M.S. in Computer Science from New York University, where I conducted research on multimodal knowledge graph representation learning. I obtained B.Eng. in Electronic Engineering from Beijing University of Posts and Telecommunications. My industrial experience includes an internship at Data, Knowledge and Intelligence group at Microsoft Research Asia.

<!-- ## News

- [Sept 1991] Attended Hogwarts
- [July 1980] Born in Godric's Hollow, West Country, England, Great Britain -->


## Education

- **University of Victoria** <br/>
Ph.D. in Computer Science, Sept 2023 - present

- **New York University** <br/>
M.S. in Computer Science, Sept 2019 - Dec 2022

- **University of California, Berkeley** <br/>
Summer Session, Jul 2016 - Aug 2016

- **Beijing University of Posts and Telecommunications** <br/>
B.Eng. in Electronic Engineering, Sept 2015 - Jun 2019

## Internship
- **Microsoft Research Asia** (full-time, onsite) <br/>
Intern at Data, Knowledge and Intelligence group, Aug 2020 - Feb 2021
  - Contributed to the research on Table2Charts.
  - Delivered Table2Charts technique to Bing search and to Excel spreadsheet intelligence.
  - Designed multilingual key-phrase extraction algorithm for questionnaire word cloud used in Forms Ideas and in Teams poll.

## Research Experience

- **Geometric Computing Lab, New York University** <br/>
Independent project, supervised by [Teseo Schneider](http://web.uvic.ca/~teseo/) and [Daniele Panozzo](https://cims.nyu.edu/gcl/daniele.html), Sept 2022 - Apr 2023 \
  Worked on GPU accelerated contact simulations in PolyFEM library.

- **CILVR Lab, New York University** <br/> 
Independent project, supervised by [Iacer Calixto](http://iacercalixto.github.io) and [Clara Vania](http://claravania.github.io), Mar 2020 - May 2021 \
  Worked on learning robust mulilingual multimodal knowledge graph representations.

- **Center for Speech and Language Technologies, Tsinghua University** <br/>
Research intern, supervised by [Dong Wang](http://wangd.cslt.org), 2019  \
  Worked on ancient Chinese poetry generation.


## Publication

<!-- [→ Full list](/projects/) -->

<ProjectCard image="/projects/01.png" >
<!-- <ProjectCard image="/projects/01.png" hideBorder=true> -->

  **MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI**
  
  Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, **Yibo Liu**, Wenhao Huang, Huan Sun, Yu Su, Wenhu Chen

  *arXiv* 2311.16502, Nov. 2023

  [[Paper](https://arxiv.org/abs/2311.16502)] [[Web Page](https://mmmu-benchmark.github.io)]

</ProjectCard>

<ProjectCard image="/projects/02.png" >

  **Endowing Language Models with Multimodal Knowledge Graph Representations**

  Ningyuan Huang, Yash R. Deshpande, **Yibo Liu**, Houda Alberts, Kyunghyun Cho, Clara Vania, Iacer Calixto

  *arXiv* 2206.13163, Jun. 2022

  [[Paper](https://arxiv.org/abs/2206.13163)]

</ProjectCard>

<ProjectCard image="/projects/03.png" >

  **VisualSem: a high-quality knowledge graph for vision and language**

Houda Alberts, Ningyuan Huang, Yash Deshpande, **Yibo Liu**, Kyunghyun Cho, Clara Vania, Iacer Calixto

*Proceedings of the 1st Workshop on Multilingual Representation Learning*, pp. 138-152, Nov. 2021. *(colocated with EMNLP 2021)*

  [[Paper](https://aclanthology.org/2021.mrl-1.13.pdf)] [[Slides](/files/MRL_slides.pdf)]

  I was the **speaker** of the presentation.

</ProjectCard>

<ProjectCard image="/projects/04.png" >

  **Table2Charts: Recommending Charts by Learning Shared Table Representations**

 Mengyu Zhou, Qingtao Li, Xinyi He, Yuejiang Li , **Yibo Liu**, Wei Ji, Shi Han, Yining Chen, Daxin Jiang, Dongmei Zhang.

*Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pp. 2389-2399, Aug. 2021

  [[Paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467279)] [[Code](https://github.com/microsoft/Table2Charts)]

</ProjectCard>

## Open-Source Projects

<ProjectCard image="/projects/1.png" >

  **Neural Topic Model Library**
  - The first Python library containing all cutting-edge neural topic models.
  - Refactored the code base framework and rewrote the interfaces, making it has compatible APIs with Gensim
LDA library.
  - The second largest contributor to the repository. Collaborated with [Leilan Zhang](https://scholar.google.com/citations?user=FDeI9yUAAAAJ&hl=zh-CN).

  [[GitHub | 384 star :star: | 77 fork](https://github.com/zll17/Neural_Topic_Models/tree/dev_b)]

</ProjectCard>

<ProjectCard image="/projects/06.png" >

  **Lock-free Linked List Library for GPUs**
  - The **first** library supporting all singly linked list operations on GPUs with CUDA.
  - Achieveing 141x speedup for insertions and deletions compared to sequential operations.
  - Individual contribution.

  [[GitHub](https://github.com/xDarkLemon/Lock_Free_Linked_List_GPU/tree/master)]

</ProjectCard>

<ProjectCard image="/projects/07.png" >

  **MMMU Benchmark for Expert AGI**
  - Collected college-level multimodal questions and conducted empirical studies on error analysis.

  [[Paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467279)] [[Web Page](https://mmmu-benchmark.github.io)]

</ProjectCard>

<ProjectCard image="/projects/10.png" >

  **COIG-PC: Chinese Open Instruction Generalist Prompt Collection**
  - Collected prompts to facilitate the fine-tuning and optimization of Chinese language models.
  
  [[Huggingface Dataset](https://huggingface.co/datasets/BAAI/COIG-PC)]

</ProjectCard>


## Blog Posts

[→ Full list](/article/)

<ProjectCard image="/projects/08.png">

  **2021科大讯飞鸟鸣识别比赛总结**

鸟类鸣叫声识别挑战赛旨在增强自动鸟类鸣叫声识别技术，预测出每个测试音频中出现的鸟类物种。比赛中，探索了多种特征提取方法、数据增强方法，对音频频谱图使用图像分类算法进行分类，探索了多种模型，包括CNN，CNN特征提取+序列模型（LSTM/Transformer），以及Vision Transformer。最终使用模型集成提升效果。

  [[full article](/article/bird_song.html)] [[code](https://github.com/xDarkLemon/BirdRec)]

</ProjectCard>

<ProjectCard image="/projects/09.png">

  **Expressive Power, Generalization, and Optimization of Graph Neural Networks: A Survey**

Summarized the theoretic frameworks of GNN’s expressive power; summarized the generalization bounds,
the generalization ability of different GNNs, and the methods to improve generalization ability; stated the
explanation of over-fitting problem and summarized the optimization methods.

  [[full article](/files/GNN_Survey.pdf)]

</ProjectCard>


## Teaching

**Teaching Assistant** for CSC503: Data Mining

Conduct laboratory sessions for a class comprising 30 students, proctor the final exam.

## Hobbies

**Cycling**: Completed a ride around Qinghai Lake covering a distance of 360 kilometers in 5 days in 2015.

**Snowboarding**: Proficient at an intermediate level.

<!-- ## Awards & Honors

### Contests

- First place in **The Hogwarts House Cup** -->


<!-- Custom style for this page -->

<style lang="stylus">

.theme-container.home-page .page
  font-size 14px
  font-family "lucida grande", "lucida sans unicode", lucida, "Helvetica Neue", Helvetica, Arial, sans-serif;
  p
    margin 0 0 0.5rem
  p, ul, ol
    line-height normal
  a
    font-weight normal
  .theme-default-content:not(.custom) > h2
    margin-bottom 0.5rem
  .theme-default-content:not(.custom) > h2:first-child + p
    margin-top 0.5rem
  .theme-default-content:not(.custom) > h3
    padding-top 4rem

  /* Override */
  .md-card
    margin-top 0.5em
    .card-image
      padding 0.2rem
      img
        max-width 120px
        max-height 120px
    .card-content p
      -webkit-margin-after 0.2em

@media (max-width: 419px)
  .theme-container.home-page .page
    p, ul, ol
      line-height 1.5

    .md-card
      .card-image
        img 
          width 100%
          max-width 400px

</style>
