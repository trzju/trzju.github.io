---
title: 重要性采样与 Off-Policy Actor-Critic
date: 2026-1-16
categories: [Math in RL]
tags: [RL, Math, Notes]
---


## 重要性采样与 Off-Policy Actor-Critic

#### 1. 核心动机：为什么需要重要性采样？

在之前的笔记 (强化学习的数学原理 Lecture 9) 中，讨论的策略梯度（Policy Gradient）通常是 **On-Policy** 的。即：要评估和优化的策略是 $\pi_\theta$，产生数据的策略也是 $\pi_\theta$。

一旦参数 $\theta$ 更新，旧的采样数据就失效了从而需要重新采样/与环境交互，这导致样本效率（Sample Efficiency）极低。

**目标**：我们希望利用一个**行为策略 (Behavior Policy) $\beta$** 产生的样本，来估计和优化**目标策略 (Target Policy) $\pi_\theta$** 的期望值 。



#### 2. 数学原理：从简单的统计例子开始

考虑一个随机变量 $X$，我们需要计算其在分布 $p_0$ 下的期望 $E_{x\sim p_0}[f(x)]$。

假设我们无法直接从 $p_0$ 采样，但能从另一个分布 $p_1$ 采样。

例子 ：

- **Target $p_0$**: $P(+1)=0.5, P(-1)=0.5 \implies E[X] = 0$.
- **Behavior $p_1$**: $P(+1)=0.8, P(-1)=0.2 \implies E[X] = 0.6$.

如果直接用 $p_1$ 的样本求平均，结果会收敛到 0.6，这对于估计 $p_0$ 的期望是错误的（有偏估计）。

重要性采样技巧 (The Trick) ： 利用期望的定义进行恒等变换：

$$
E_{x\sim p_0}[f(x)] = \sum_x p_0(x)f(x) = \sum_x p_1(x) \frac{p_0(x)}{p_1(x)} f(x) = E_{x\sim p_1}\left[ \frac{p_0(x)}{p_1(x)} f(x) \right]
$$

其中 $\frac{p_0(x)}{p_1(x)}$ 被称为**重要性权重 (Importance Weight)**。

- **直觉**：如果一个样本 $x$ 在 $p_0$ 中出现的概率比在 $p_1$ 中大（权重 $>1$），说明它对 $p_0$ 很重要，我们需要在计算期望时“放大”该样本的贡献；反之则“缩小” 。

通过加权平均，即使样本来自 $p_1$，我们也能**无偏地**估计 $p_0$ 的期望：

$$
\frac{1}{N} \sum_{i=1}^N \frac{p_0(x_i)}{p_1(x_i)} x_i \xrightarrow{N\to\infty} E_{x\sim p_0}[X]
$$

#### 3. 离策略策略梯度定理 (Off-Policy Policy Gradient)

将上述数学原理应用到强化学习中。

- **Target Policy**: $\pi(a|s,\theta)$ (需要优化的)
- **Behavior Policy**: $\beta(a|s)$ (负责与环境交互产生轨迹的)

我们希望最大化的目标函数 $J(\theta)$ 是基于目标策略 $\pi$ 的价值期望。 在 On-Policy 中，梯度为 $E_{\pi}[\nabla \ln \pi(a|s) Q^\pi(s,a)]$。 在 Off-Policy 中，引入重要性权重 $\rho(s,a) = \frac{\pi(a|s)}{\beta(a|s)}$，梯度变为 ：

$$
\nabla_\theta J(\theta) = E_{s\sim\rho^\beta, a\sim\beta} \left[ \frac{\pi(a|s,\theta)}{\beta(a|s)} \nabla_\theta \ln \pi(a|s,\theta) Q^\pi(s,a) \right]
$$

**注意**：

这里的分布 $s \sim \rho^\beta$ 表示状态是由行为策略 $\beta$ 访问到的。理论上状态分布的 mismatch 也需要校正，但通常在近似算法中会**忽略状态分布的差异**（或者认为 $\pi$ 和 $\beta$ 的状态访问分布差异不大），只校正动作概率的差异。

#### 4. Off-Policy Actor-Critic 算法流程

基于上述定理，便可以构造 Off-Policy AC 算法。与标准 AC 的主要区别在于 update rule 中加入了重要性权重。



算法循环 ：

1. **采样**: 使用行为策略 $\beta$ 生成动作 $a_t$，观测 $r_{t+1}, s_{t+1}$。

2. **计算 TD Error (Advantage)**:

   使用 Critic 估计价值，计算优势函数（形式上由 TD error 代替）：

   $$\delta_t = r_{t+1} + \gamma V(s_{t+1}) - V(s_t)$$

3. **Actor 更新 (Policy Update)**:

   $$
   \theta_{t+1} = \theta_t + \alpha_\theta \underbrace{\frac{\pi(a_t|s_t, \theta_t)}{\beta(a_t|s_t)}}_{\text{Importance Weight}} \delta_t \nabla_\theta \ln \pi(a_t|s_t, \theta_t)
   $$

   - 直觉：如果 $\pi$ 认为动作 $a_t$ 的概率应该比 $\beta$ 实际采样的概率大，那么这个梯度的更新步长就会被放大。

4. **Critic 更新 (Value Update)**:

   Critic 的更新同样需要加权，因为它是在拟合 $\pi$ 的价值函数 $V^\pi$，但数据来自 $\beta$：

   $$
   w_{t+1} = w_t + \alpha_w \frac{\pi(a_t|s_t, \theta_t)}{\beta(a_t|s_t)} \delta_t \nabla_w V(s_t, w_t)
   $$

#### 5. 关键性质

- **基线不变性 (Baseline Invariance)** ： 即使引入了重要性权重，减去基线（Baseline）$b(s)$ 依然不会改变梯度的期望，但能有效降低方差。通常取 $b(s) = V(s)$。
  $$
  \nabla J = E \left[ \frac{\pi}{\beta} \nabla \ln \pi (Q^\pi(s,a) - V^\pi(s)) \right]
  $$

- **方差问题 (Variance Issue)**：

  重要性采样最大的问题是方差。如果 $\pi$ 和 $\beta$ 差异巨大，比值 $\frac{\pi}{\beta}$ 可能非常大（爆炸）或接近 0。

  - 如果 $\frac{\pi}{\beta} \gg 1$，梯度更新会极不稳定。
  - 这就是为什么在 PPO (Proximal Policy Optimization) 等现代算法中，会通过 `clip` 操作强制限制这个比值的范围（例如限制在 $[0.8, 1.2]$ 之间）。

- **确定性策略 (Deterministic Policy)** ： 如果策略是确定性的 $a = \mu(s)$（如 DDPG 算法），它天然是 Off-Policy 的，因为确定性策略梯度的计算不需要对动作分布积分（也就规避了动作的重要性采样比值问题），这在连续动作控制中非常高效。
