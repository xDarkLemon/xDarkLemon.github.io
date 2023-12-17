---
pageClass: home-page
# some data for the components

name: Yibo Liu
profile: /myprofile.jpeg

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


cv: https://en.wikipedia.org/wiki/Harry_Potter
bio: Computer Science
email: yl6769 (at) nyu (dot) edu
---

<ProfileSection :frontmatter="$page.frontmatter" />

## About

I am Yibo Liu, a first year Ph.D. student in Computer Science at University of Victoria, supervised by Prof. [Teseo Schneider](http://web.uvic.ca/~teseo/). Currently I am working on geometric processing and physics simulations. Prior to this, I obtained my M.S. in Computer Science from New York University, where I conducted research on multimodal knowledge graph representation learning. I obtained B.Eng. in Electronic Engineering from Beijing University of Posts and Telecommunications. My research interests span on natural language process and computer graphics. My industrial experience includes an internship at Data, Knowledge and Intelligence group at Microsoft Research Asia.

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

## Internships
- **Microsoft Research Asia** <br/>
Intern at Data, Knowledge and Intelligence group, Aug 2020 - Feb 2021

- **Tsinghua University** <br/>
Research Intern at Center for Speech and Language Technologies, 2019

## Publication


[→ Full list](/projects/)

<ProjectCard image="/projects/1.png" hideBorder=true>

  **MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI**
  
  Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, **Yibo Liu**, Wenhao Huang, Huan Sun, Yu Su, Wenhu Chen

  *arXiv* 2311.16502, Nov. 2023

  [[arXiv](https://arxiv.org/abs/2311.16502)] [[web](https://mmmu-benchmark.github.io)]

</ProjectCard>

<ProjectCard image="/projects/1.png" hideBorder=true>

  **Endowing Language Models with Multimodal Knowledge Graph Representations**

  Ningyuan Huang, Yash R. Deshpande, **Yibo Liu**, Houda Alberts, Kyunghyun Cho, Clara Vania, Iacer Calixto

  *arXiv* 2206.13163, Jun. 2022

  [[arXiv](https://arxiv.org/abs/2206.13163)]

</ProjectCard>

<ProjectCard image="/projects/1.png" hideBorder=true>

  **VisualSem: a high-quality knowledge graph for vision and language**

Houda Alberts, Ningyuan Huang, Yash Deshpande, **Yibo Liu**, Kyunghyun Cho, Clara Vania, Iacer Calixto

*Proceedings of the 1st Workshop on Multilingual Representation Learning*, pp. 138-152, Nov. 2021.

  [[paper](https://aclanthology.org/2021.mrl-1.13.pdf)]

</ProjectCard>

<ProjectCard image="/projects/1.png" hideBorder=true>

  **Table2Charts: Recommending Charts by Learning Shared Table Representations**

 Mengyu Zhou, Qingtao Li, Xinyi He, Yuejiang Li , **Yibo Liu**, Wei Ji, Shi Han, Yining Chen, Daxin Jiang, Dongmei Zhang.

*Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pp. 2389-2399, Aug. 2021

  [[paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467279)]

</ProjectCard>

## Open-Source Contribution

<ProjectCard image="/projects/1.png" hideBorder=true>

  **Neural Topic Model Library**

  [[code](https://github.com/zll17/Neural_Topic_Models/tree/dev_b)]

</ProjectCard>


<ProjectCard image="/projects/1.png" hideBorder=true>

  **MMMU Benchmark for Expert AGI**

  [[paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467279)] [[web](https://mmmu-benchmark.github.io)]

</ProjectCard>

<ProjectCard image="/projects/1.png" hideBorder=true>

  **COIG-PC: Chinese Open Instruction Generalist Prompt Collection**

  [[dataset](https://huggingface.co/datasets/BAAI/COIG-PC)]

</ProjectCard>

<ProjectCard image="/projects/1.png" hideBorder=true>

  **Lock-free Linked List Library for GPUs**

  [[code](https://github.com/xDarkLemon/Lock_Free_Linked_List_GPU/tree/master)]

</ProjectCard>

## Teaching

Teaching Assistant for CSC503: Data Mining

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
