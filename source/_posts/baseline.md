---
title: 方差缩减与 Baseline 方法
date: 2026-1-16
categories: [Math in RL]
tags: [RL, Math, Notes]
---


## 方差缩减与 Baseline 方法

#### 1. 核心动机：策略梯度的方差问题

在原始的策略梯度公式 $\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \nabla_\theta \log \pi_\theta(\tau) r(\tau)$ 中，我们面临一个严重的问题：**高方差 (High Variance)**。--  [**CS285 Lecture 5 Part 3**](https://youtu.be/VgdSubQN35g?si=cktuhrM7gOhada2N)

- **直觉理解**：

  假设在一个环境中，所有的轨迹回报 $r(\tau)$ 都是正数（例如 $[100, 101, 102]$）。

  虽然有些轨迹比另一些好，但策略梯度会试图提高**所有**轨迹的概率（因为 $r(\tau) > 0$），仅仅是提高的幅度不同 (最大似然估计)。这使得梯度估计非常嘈杂且不稳定。

  理想情况下，我们希望：对于好于平均的轨迹，增加概率；对于差于平均的轨迹，降低概率 (加权最大似然估计)。

- **数学目标**：

  我们希望找到一种方法修改 $r(\tau)$，使得梯度的期望（方向）不变，但方差降低。

#### 2. 数学原理：Baseline 的引入与无偏性证明

为了解决上述问题，引入一个**标量** $b$（Baseline），并将梯度更新修改为：
$$
\nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \nabla_\theta \log p_\theta(\tau) [r(\tau) - b]
$$
**关键问题：这样做会改变梯度的方向吗？**

答案是：**不会。减去基线是无偏的 (Unbiased)。**

**证明：**

证明这一项的期望为 0：$E_{\tau \sim p_\theta(\tau)} [\nabla_\theta \log p_\theta(\tau) \cdot b] = 0$。

利用强化学习中常见的**对数导数**技巧（Log-derivative trick）$\nabla_\theta \log p_\theta(\tau) = \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)}$：
$$
\begin{aligned} E[\nabla_\theta \log p_\theta(\tau) b] &= \int p_\theta(\tau) \nabla_\theta \log p_\theta(\tau) b d\tau \\ &= \int p_\theta(\tau) \frac{\nabla_\theta p_\theta(\tau)}{p_\theta(\tau)} b d\tau \\ &= b \int \nabla_\theta p_\theta(\tau) d\tau \\ &= b \nabla_\theta \underbrace{\int p_\theta(\tau) d\tau}_{=1} \\ &= b \cdot 0 = 0 \end{aligned}
$$
由于概率密度函数的积分为 1，其梯度为 0。因此，任何不依赖于动作 $a$ 的基线 $b$（甚至可以是依赖于状态的函数 $b(s)$）都不会改变梯度的期望值，只会改变方差。

#### 3. 推导最优 Baseline (Analyzing Variance)

既然任何 $b$ 都是无偏的，那么**哪一个 $b$ 能让方差最小？**

详细推导见 [**CS285 Lecture 5 Part 3**](https://youtu.be/VgdSubQN35g?si=cktuhrM7gOhada2N) 。

定义梯度估计量为 $X$，我们需要最小化 $Var[X]$。

$$Var[X] = E[X^2] - (E[X])^2$$

注意，$(E[X])^2$ 这一项就是真实梯度的平方，它不受 $b$ 的影响（因为前面证明了 $b$ 是无偏的）。因此，最小化方差等价于最小化第二矩 $E[X^2]$。

记 $g(\tau) = \nabla_\theta \log p_\theta(\tau)$。我们的目标是最小化关于 $b$ 的函数：
$$
J(b) = E_{\tau \sim p_\theta} [ (g(\tau)(r(\tau) - b))^2 ]
$$
**求解步骤：**

1. 对 $b$ 求导并令导数为 0：
   $$
   \frac{d}{db} E [ g(\tau)^2 (r(\tau) - b)^2 ] = 0
   $$

2. 交换求导与期望（由线性性质）：

   $$
   E [ g(\tau)^2 \cdot 2(r(\tau) - b) \cdot (-1) ] = 0
   $$

3. 整理方程：

   $$
   -2 E [ g(\tau)^2 r(\tau) ] + 2b E [ g(\tau)^2 ] = 0
   $$

4. 解出 $b^*$：

   $$
   b^* = \frac{E [ g(\tau)^2 r(\tau) ]}{E [ g(\tau)^2 ]}
   $$

**结论：**

最优的 Baseline $b^*$ 是**回报 $r(\tau)$ 的加权平均**，权重是梯度的平方模长 $g(\tau)^2$。

这在数学上非常优美，但在实际中计算 $g(\tau)^2$（梯度的平方）计算量较大且带有噪声，所以通常直接采用无加权平均值。

#### 4. 在强化学习中的实际应用

虽然推导出了最优 Baseline $b^*$，但在实际的 Deep RL 算法中，我们通常使用更简单的近似，通常效果已经足够好：

1. **平均回报 (Average Reward)**：

   最简单的 Baseline 就是所有轨迹回报的平均值：

   $$
   b \approx \frac{1}{N} \sum_{i=1}^N r(\tau)
   $$

   > It's good enough. 

2. **状态价值函数 (State Value Function)**：

   为了进一步减小方差并利用**因果性**（Causality），我们将 $r(\tau)$ 替换为从当前时刻开始的**累积回报**（Reward-to-go）$\hat{Q}_{i,t}$。

   > 因果性不同于马尔可夫性质，所有问题均满足因果性。即过去以及获得的奖励是无法被改变的。
   >
   > 这里提到的累积回报即未来的奖励累积，不计算过去已经获得的奖励。
   
   此时，最好的 Baseline 是依赖于状态 $s_t$ 的价值期望：

   $$
   b(s_t) = V^\pi(s_t) = E[ \sum_{t'=t}^T r(s_{t'}, a_{t'}) | s_t ]
   $$
   
   这就引出 **Advantage (优势函数)** 形式：
   
   $$
   \nabla_\theta J(\theta) \approx \frac{1}{N} \sum_{i=1}^N \sum_{t=1}^T \nabla_\theta \log \pi_\theta(a_{i,t}|s_{i,t}) \underbrace{( \hat{Q}_{i,t} - V^\pi(s_{i,t}) )}_{\text{Advantage } A^\pi(s,a)}
   $$

#### 总结

- **问题**：PG 方差大。
- **方法**：减去 Baseline。
- **原理**：利用 $\nabla \int p = 0$ 的性质，保证无偏。
- **最优解**：理论上的最优 Baseline 是回报的梯度加权平均，但实际中常使用状态价值函数 $V(s)$ 近似，这构成了 Actor-Critic 方法的基础。