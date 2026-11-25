---
title: 把Bayesian Inference用在UQ或许是个好想法，但可能不够。
date: 2026-10-23
summary: '分享发表在会议ICLR 2026上的一篇关于Prompt的UQ的文章（Brendan Leigh Ross, Noël Vouitsis, et al. *Textual Bayes: Quantifying Prompt Uncertainty in LLM-Based Systems.* ICLR 2026.），聊聊他的想法以及问题'
tags: [论文分享]
draft: true
---
## 0、基础知识
假设读者已经了解贝叶斯推断的一些技术，以及粗略知道大模型是怎么工作的。先说说Propmt 的 UQ 吧。

UQ 就是 Uncertainty Quantification 的缩写，即不确定性量化。顾名思义，这个工作主要研究的就是Prompt的不确定性量化

乍一看，在现在Agent（*Agent 的输入不像传统一段 prompt了，而是：用户的初始prompt+每一步的工具返回+环境状态的变化+ Agent 自己生成的中间推理......）爆发横行的时代（尽管似乎只过去不久...），Prompt的“地位”好像越来越低，量化它好像显得也不太必要了。



这在之前是相当有用的

## 一、概览
为了方便读者阅读，我先在这里大概介绍一下这篇文章的主要内容。