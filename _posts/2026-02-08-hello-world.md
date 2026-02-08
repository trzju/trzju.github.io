---
layout: default
title: 博客测试：RL算法与代码高亮
date: 2026-02-08
---

<script src="https://polyfill.io/v3/polyfill.min.js?features=es6"></script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>

## 1. 代码高亮测试 (Python)

这是我在 Mujoco 环境中配置的测试代码：

```python
import gym
import mujoco_py

env = gym.make('Humanoid-v2')
env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # 随机动作
env.close()
