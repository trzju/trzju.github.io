---
title: 4. 价值函数近似与策略梯度法
date: 2026-1-05
categories: [Reinforcement Learning]
tags: [RL, Notes]
---

## Value Function Approximation

从 tabular 到 function approximation 的核心动机：  
状态动作空间太大，表格无法存、无法泛化。

### 1) 目标函数与分布选择

用参数化函数 $\hat v(s,w)$ 逼近 $v_\pi(s)$，常用目标：
$$
J(w)=\mathbb{E}_{S\sim d}\big[(v_\pi(S)-\hat v(S,w))^2\big]
$$

- $w$：可学习参数向量。
- $\hat v(s,w)$：近似器输出。
- $d$：状态采样分布（可取均匀分布或策略平稳分布 $d_\pi$）。

### 2) MC + 函数逼近

若用 Monte Carlo 目标 $g_t$（trajectory return）：
$$
w_{t+1}=w_t+\alpha_t\big[g_t-\hat v(s_t,w_t)\big]\nabla_w\hat v(s_t,w_t)
$$

### 3) TD + 函数逼近（semi-gradient）

$$
w_{t+1}=w_t+\alpha_t\big[r_{t+1}+\gamma\hat v(s_{t+1},w_t)-\hat v(s_t,w_t)\big]\nabla_w\hat v(s_t,w_t)
$$

- 方括号内是 TD error 的函数逼近版。
- 这是“TD target + 局部梯度”结构。

### 4) 线性逼近与 tabular 的关系

线性形式：
$$
\hat v(s,w)=\phi(s)^T w,\qquad \nabla_w\hat v(s,w)=\phi(s)
$$

- $\phi(s)$：特征向量（feature）。
- 若 $\phi(s)$ 取 one-hot，对应回到 tabular 表示。

### 5) Sarsa/Q-learning + 函数逼近

令 $\hat q(s,a,w)$ 为动作价值近似器。

Sarsa：
$$
w_{t+1}=w_t+\alpha_t\big[r_{t+1}+\gamma\hat q(s_{t+1},a_{t+1},w_t)-\hat q(s_t,a_t,w_t)\big]\nabla_w\hat q(s_t,a_t,w_t)
$$

Q-learning：
$$
w_{t+1}=w_t+\alpha_t\big[r_{t+1}+\gamma\max_a\hat q(s_{t+1},a,w_t)-\hat q(s_t,a_t,w_t)\big]\nabla_w\hat q(s_t,a_t,w_t)
$$

### 6) Deep Q-learning（DQN）关键机制

损失：
$$
\mathcal{L}(w)=\mathbb{E}_{(s,a,r,s')\sim\mathcal{B}}\left[\left(y-r-\hat q(s,a,w)\right)^2\right],\quad
y=r+\gamma\max_{a'}\hat q(s',a',w^-)
$$

- $w$：主网络参数（online/main net）。
- $w^-$：目标网络参数（target net），周期性从 $w$ 同步。
- $\mathcal{B}$：replay buffer。

两大稳定技巧：

1. **Target network**：让 target 慢变化，减少“追着自己移动目标”。
2. **Experience replay**：打破样本强相关，近似 i.i.d. 训练条件。

---

## Policy Gradient (or policy function approximation)

从 value-based 切到 policy-based：直接优化策略参数 $\theta$。

### 1) 参数化策略

$$
\pi_\theta(a|s)
$$

- $\theta$：策略网络参数。
- $\pi_\theta(a|s)$：在状态 $s$ 选动作 $a$ 的概率密度/质量。

### 2) 目标函数（metrics）

常见写法（折扣场景）：
$$
J(\theta)=\mathbb{E}_{s\sim d^{\pi_\theta},a\sim\pi_\theta}[\,\cdot\,]
$$
具体“点号”可以是平均 state value、平均 reward 等等。

### 3) Policy Gradient Theorem（核心形态）

$$
\nabla_\theta J(\theta)
=\mathbb{E}_{s\sim d^{\pi_\theta},a\sim\pi_\theta}\Big[q^{\pi_\theta}(s,a)\nabla_\theta\log\pi_\theta(a|s)\Big]
$$

这条公式是主线：  
**对策略求导**，转化成“score function（$\nabla\log\pi$）乘以价值信号”的期望。

### 4) 随机梯度上升（REINFORCE 形式）

$$
\theta_{t+1}=\theta_t+\alpha_t\,\hat q_t(s_t,a_t)\,\nabla_\theta\log\pi_{\theta_t}(a_t|s_t)
$$

- 若 $\hat q_t$ 用完整 MC return（如 $G_t$）估计，就是 REINFORCE。
- 本质是 on-policy（采样分布来自当前策略）。

### 5) baseline 降方差

利用不变性：
$$
\mathbb{E}\left[(q(s,a)-b(s))\nabla_\theta\log\pi_\theta(a|s)\right]
$$
与原梯度同均值，但可显著降方差。—— CS 285 lecture5.part3. 

- $b(s)$：只依赖状态、与动作无关的 baseline。
- 常用选择：$b(s)=v^\pi(s)$，于是权重变成 advantage：
$$
A^\pi(s,a)=q^\pi(s,a)-v^\pi(s)
$$
