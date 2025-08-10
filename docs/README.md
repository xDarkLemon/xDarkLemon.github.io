---
pageClass: home-page
# some data for the components

name: Yibo Liu
profile: /myprofile2.jpeg
bio: CS PhD Candidate @ UVic
email: liuyibo (at) uvic (dot) ca

socials:
  - title: google-scholar
    icon: "/icons/google-scholar.svg"
    link: https://scholar.google.com/citations?user=FQExy98AAAAJ&hl=en&oi=ao

  - title: github
    icon: "/icons/github.svg"
    link: https://github.com/xDarkLemon

  - title: linkedin
    icon: "/icons/linkedin.svg"
    link: https://www.linkedin.com/in/yibo-liu-5a60a21b5/
    
  - title: twitter
    icon: "/icons/twitter.svg"
    link: https://twitter.com/_liuyibo

  - title: CV
    icon: "/icons/cv.svg"
    link: /files/Yibo_Liu_CV_GM.pdf

actions:  # buggy? not working?
  - text: Projects
    link: /projects/
---

<ProfileSection :frontmatter="$page.frontmatter" />

## About

I am Yibo Liu, a second year Ph.D. student in Computer Science at University of Victoria ([UVIC](https://www.uvic.ca/)), working with [Prof. Dr. Teseo Schneider](http://web.uvic.ca/~teseo/) . Prior to this, I obtained my M.S. in Computer Science from New York University ([NYU](https://www.nyu.edu/)) and B.Eng. in Electronic Engineering from Beijing University of Posts and Telecommunications ([BUPT](https://www.bupt.edu.cn/)). Previously, I have interned at Microsoft Research Asia ([MSRA](https://www.microsoft.com/en-us/research/lab/microsoft-research-asia/)). 

My current research interest focuses on how LLMs and neural networks enhance the intelligence and expressivity of physical-based simulations.

<!-- ## News

- [Sept 1991] Attended Hogwarts
- [July 1980] Born in Godric's Hollow, West Country, England, Great Britain -->


<!-- ## Education

- **University of Victoria**, BC, Canada <br/> 
Ph.D. in Computer Science, Sept 2023 - present
Supervisor: [Teseo Schneider](http://web.uvic.ca/~teseo/)

- **New York University**, NY, USA <br/>
M.S. in Computer Science, Sept 2019 - Dec 2022

- **Beijing University of Posts and Telecommunications**, Beijing, China <br/>
B.Eng. in Electronic Engineering, Sept 2015 - Jun 2019 -->

## Experience
- **Univerisity of Victoria** | Sept 2023 - Present  <br/>
PhD Candidate, Computer Science \
  <!-- Conducting research on neural kinematic bases for fluid simulation to accelerate physically-based fluid dynamics, and on multimodal language-based models for crowd simulation that leverage large language models for semantic reasoning and agent behavior modeling. -->

- **Geometric Computing Lab, New York University** | 2022 - 2023 <br/>
Independent Study, supervised by [Prof. Dr. Teseo Schneider](http://web.uvic.ca/~teseo/) and [Prof. Dr. Daniele Panozzo](https://cims.nyu.edu/gcl/daniele.html) \
  Conducting research on GPU acceleration for contact simulations in [PolyFEM](http://github.com/polyfem/polyfem) library.

- **Microsoft Research Asia** | Aug 2020 - Feb 2021  <br/>
R&D Intern at *Data, Knowledge and Intelligence* group, full-time, onsite \
  Conducted research on recommending charts from tables (Table2Charts) using reinforcement learning, delivering the technique to Bing Search and Excel Spreadsheet Intelligence, and developed a multilingual key-phrase extraction algorithm used in Forms Ideas and Teams polls.

- **CILVR Lab, New York University** | Mar 2020 - May 2021 <br/> 
Independent Study, supervised by [Prof. Dr. Iacer Calixto](http://iacercalixto.github.io) and [Dr. Clara Vania](http://claravania.github.io)\
  Conducting research on learning robust mulilingual multimodal knowledge graph representations.

- **Center for Speech and Language Technologies, Tsinghua University** | 2018 - 2019<br/>
Research intern, supervised by [Prof. Dr. Dong Wang](http://wangd.cslt.org)  \
  Conducting research on poetry generation with rythm (YunLv) as soft constraint.

## Publication

<!-- [→ Full list](/projects/) -->

<ProjectCard image="/projects/crowds.png">

  **An LLM Agent Based Emotion Model for Crowds Simulation**
  
  [[Paper](TODO)] [[Demo](TODO)]

</ProjectCard>

<ProjectCard image="/projects/question.png">

  ***Cool Stuff In Submission***
  
  [[Paper](TODO)] [[Demo](TODO)]

</ProjectCard>


<ProjectCard image="/projects/01.png" >
<!-- <ProjectCard image="/projects/01.png" hideBorder=true> -->

  **MMMU: A Massive Multi-discipline Multimodal Understanding and Reasoning Benchmark for Expert AGI**
  
  Xiang Yue, Yuansheng Ni, Kai Zhang, Tianyu Zheng, Ruoqi Liu, Ge Zhang, Samuel Stevens, Dongfu Jiang, Weiming Ren, Yuxuan Sun, Cong Wei, Botao Yu, Ruibin Yuan, Renliang Sun, Ming Yin, Boyuan Zheng, Zhenzhu Yang, **Yibo Liu**, Wenhao Huang, Huan Sun, Yu Su, Wenhu Chen

**[CVPR 2024 Best Paper Nomination]** *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*, pp. 9556-9567, Jun. 2024

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

  [[Paper](https://aclanthology.org/2021.mrl-1.13.pdf)] 

</ProjectCard>

<ProjectCard image="/projects/04.png" >

  **Table2Charts: Recommending Charts by Learning Shared Table Representations**

 Mengyu Zhou, Qingtao Li, Xinyi He, Yuejiang Li , **Yibo Liu**, Wei Ji, Shi Han, Yining Chen, Daxin Jiang, Dongmei Zhang.

*Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery and Data Mining*, pp. 2389-2399, Aug. 2021

  [[Paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467279)] [[Code](https://github.com/microsoft/Table2Charts)]

</ProjectCard>

## Open-Source Contribution
<ProjectCard image="/projects/ntm.png" >

  **Neural Topic Model Library**
  - The first Python library containing all cutting-edge neural topic models, collaborated with [Leilan Zhang](https://scholar.google.com/citations?user=FDeI9yUAAAAJ&hl=zh-CN).
  - Refactored the codebase framework and rewrote the interfaces to provide APIs compatible with the Gensim LDA library.

  [[GitHub | 431 star :star: | 80 fork](https://github.com/zll17/Neural_Topic_Models/tree/dev_b)]

</ProjectCard>

<ProjectCard image="/projects/polyfem.png" >

  **PolyFEM**
 - Developed GPU acceleration for contact simulations.
 - Profiled contact solvers in the PolyFEM library, identified performance bottlenecks.

  [[GitHub | 562 star :star: | 88 fork](TODO)]

</ProjectCard>

## Projects

[→ Full list](/projects/)

<ProjectCard image="/projects/gpu.png" >

  **Lock-free Linked List Library for GPUs**
  - This project develops a CUDA-based singly linked list library, with all operations running on GPUs to fully exploit their parallelism.
  - It introduces the **first** lock-free linked list algorithm on CUDA, adapted from prior implementations on multi-core CPUs.

  [[GitHub](https://github.com/xDarkLemon/concurrent_linked_list_cuda)] [[Report](/files/gpu_final_report.pdf)]

</ProjectCard>

<ProjectCard image="/projects/deform.png" >

  **A Python Implementation for *Harmonic Coordinates for Character Articulation***
  - Implemented a Python-based framework for harmonic coordinates to enable smooth and natural deformation in character articulation. 
  - The system allows intuitive control of complex mesh movements while preserving geometric consistency.

  [[GitHub](https://github.com/xDarkLemon/Harmonic-Coordinates-for-Character-Articulation)]

</ProjectCard>

<ProjectCard image="/projects/09.png">

  **Expressive Power, Generalization, and Optimization of Graph Neural Networks: A Survey**
 - Summarized theoretical frameworks on GNN expressive power and generalization, offering a structured understanding of their strengths and limitations.
 - Examined overfitting challenges and outlined optimization strategies to improve GNN performance in practice.

<!-- Summarized the theoretic frameworks of GNN’s expressive power; summarized the generalization bounds,
the generalization ability of different GNNs, and the methods to improve generalization ability; stated the
explanation of over-fitting problem and summarized the optimization methods. -->

  [[Report](/files/GNN_Survey.pdf)]

</ProjectCard>



<!-- <ProjectCard image="/projects/10.png" >

  **COIG-PC: Chinese Open Instruction Generalist Prompt Collection**
  - Collected prompts to facilitate the fine-tuning and optimization of Chinese language models.
  
  [[Huggingface Dataset](https://huggingface.co/datasets/BAAI/COIG-PC)]

</ProjectCard> -->

<!-- <ProjectCard image="/projects/07.png" >

  **MMMU Benchmark for Expert AGI**
  - Collected college-level multimodal questions and conducted empirical studies on error analysis.

  [[Paper](https://dl.acm.org/doi/pdf/10.1145/3447548.3467279)] [[Web Page](https://mmmu-benchmark.github.io)]

</ProjectCard> -->

<!-- <ProjectCard image="/projects/10.png" >

  **COIG-PC: Chinese Open Instruction Generalist Prompt Collection**
  - Collected prompts to facilitate the fine-tuning and optimization of Chinese language models.
  
  [[Huggingface Dataset](https://huggingface.co/datasets/BAAI/COIG-PC)]

</ProjectCard>


## Blog Posts

<!-- [→ Full list](/article/) -->

<!-- <ProjectCard image="/projects/08.png">

  **2021科大讯飞鸟鸣识别比赛总结**

鸟类鸣叫声识别挑战赛旨在增强自动鸟类鸣叫声识别技术，预测出每个测试音频中出现的鸟类物种。比赛中，探索了多种特征提取方法、数据增强方法，对音频频谱图使用图像分类算法进行分类，探索了多种模型，包括CNN，CNN特征提取+序列模型（LSTM/Transformer），以及Vision Transformer。最终使用模型集成提升效果。

  [[full article](/article/bird_song.html)] [[code](https://github.com/xDarkLemon/BirdRec)]

</ProjectCard> -->

## Talks
<!-- **Apr 2025** I gave a guest talk *["Neural Kinematic Fields for Fluids"]()* for the course CSC 586 Geometric Modeling, University of Victoria. -->

**Oct 2021** I presented our paper *["VisualSem: a high-quality knowledge graph for vision and language"](/files/MRL_slides.pdf)* at [EMNLP Workshop on Multilingual Representation Learning 2021](https://www.aclweb.org/portal/content/1st-workshop-multilingual-representation-learning-mrl-emnlp).


## Peer Review Service
**AAAI Conference on Artificial Intelligence (AAAI)**, 2025

**International Conference on Learning Representations (ICLR)**, 2025

**Transactions on Visualization and Computer Graphics (TVCG)**, 2024

## Volunteer
**SIGGRAPH Asia 2024** Student Volunteer | Tokyo, Japan | Dec 2024

## Teaching Assistantship
**Introduction to Computer Graphics** | University of Victoria | 2025 summer

**Data Mining** | University of Victoria | 2025 spring, 2024 winter, 2023 fall

**Introduction to C++** | University of Victoria | 2024 fall

**Software Architecture** | University of Victoria | 2024 fall

## Skills
- Python, PyTorch, CUDA, C++, C#
- Blender, Unity, ParaView

## Blender Art Gallery

<!-- ## Hobbies

**Cycling**: Cycled around Qinghai Lake (360 kilometers) in 5 days.

**Snowboarding**: Intermediate level.

**Bouldering**: Beginner level. -->

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
