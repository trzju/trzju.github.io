---
title: 5. Actor-critic 方法
date: 2026-1-06
categories: [Reinforcement Learning]
tags: [RL, Notes]
---

## Actor-Critic

一句话：**Actor 用 policy gradient 更新策略；Critic 估计价值/优势给 Actor 当学习信号。**

### 1) 最简单 Q Actor-Critic（QAC）

Actor 更新：
$$
\theta_{t+1}=\theta_t+\alpha_\theta \nabla_\theta\log\pi_{\theta_t}(a_t|s_t)\,\hat q(s_t,a_t,w_t)
$$

Critic（可用 Sarsa 风格 TD）：
$$
w_{t+1}=w_t+\alpha_w\delta_t^q\nabla_w\hat q(s_t,a_t,w_t)
$$
其中
$$
\delta_t^q=r_{t+1}+\gamma\hat q(s_{t+1},a_{t+1},w_t)-\hat q(s_t,a_t,w_t)
$$

### 2) Advantage Actor-Critic（A2C）

用状态价值 baseline：
$$
\delta_t=r_{t+1}+\gamma\hat v(s_{t+1},w_t)-\hat v(s_t,w_t)\approx A^\pi(s_t,a_t)
$$

Actor：
$$
\theta_{t+1}=\theta_t+\alpha_\theta \nabla_\theta\log\pi_{\theta_t}(a_t|s_t)\,\delta_t
$$

Critic：
$$
w_{t+1}=w_t+\alpha_w\delta_t\nabla_w\hat v(s_t,w_t)
$$

### 3) Off-policy Actor-Critic（importance sampling）

若样本由行为策略 $\beta$ 生成，而目标策略是 $\pi_\theta$，用比率
$$
\rho_t=\frac{\pi_\theta(a_t|s_t)}{\beta(a_t|s_t)}
$$
修正：
$$
\theta_{t+1}=\theta_t+\alpha_\theta\,\rho_t\,\delta_t\,\nabla_\theta\log\pi_{\theta_t}(a_t|s_t)
$$

### 4) Deterministic Policy Gradient（DPG）

确定性策略：$a=\mu_\theta(s)$。  
梯度形式：
$$
\nabla_\theta J(\theta)=\mathbb{E}_{s\sim\rho^\beta}\left[\nabla_a q^\mu(s,a)\big|_{a=\mu_\theta(s)}\nabla_\theta\mu_\theta(s)\right]
$$

这就是 deterministic actor-critic / DDPG 系列的核心。

---

### L5-L10

1. MC：用完整回报做估计，简单直接，但方差高、更新慢。  
2. SA：给出“随机迭代也可收敛”的理论底座。  
3. TD：把 MC 的完整回报替换成 bootstrap target，增量高效。  
4. Value Function Approximation：把“表格值”推广为“参数函数”，走向深度方法。  
5. Policy Gradient：直接优化策略分布参数。  
6. Actor-Critic：把 PG（actor）和 TD/value estimation（critic）合并，兼顾效率与稳定性。

