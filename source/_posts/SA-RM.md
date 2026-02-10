---
title: 随机近似 (Stochastic Approximation) 与 RM 算法
date: 2026-1-16
categories: [Math in RL]
tags: [RL, Math, Notes]
---

## 随机近似 (Stochastic Approximation) 与 RM 算法

$New \leftarrow Old + \alpha (Target - Old)$ 

#### 1. 直觉引入：从“增量式求平均”说起

假设有一个随机变量 $X$，想求它的期望 $E[X]$。

通常做法是收集 $N$ 个样本取平均：$\bar{x} = \frac{1}{N}\sum x_i$。

但在强化学习中，数据是流式到达的，我们不想存下所有历史数据。

于是可以把求平均写成**增量迭代 (Incremental)** 的形式 ：


$$
w_{k+1} = w_k - \frac{1}{k}(w_k - x_k)
$$

或者引入更一般的步长 $\alpha_k$ ：


$$
w_{k+1} = w_k + \alpha_k (x_k - w_k)
$$

**直觉解释**：

- $x_k - w_k$ 是 **误差 (Error)**：当前样本 $x_k$ 和当前估计 $w_k$ 的差。
- 让估计值 $w$ 朝着样本 $x$ 的方向移动一小步 $\alpha$。
- 虽然单个 $x_k$ 有噪声，但只要 $\alpha_k$ 逐渐变小，$w_k$ 最终会收敛到真实期望 $E[X]$。

这其实就是 **Robbins-Monro 算法** 的一个特例。

#### 2. 数学原理：Robbins-Monro (RM) 算法


**RM 算法要解决的核心问题** ： 找到方程 $g(w) = 0$ 的根 $w^*$。


**困难点** ：
- 函数 $g(w)$ 的解析式是未知的（Black-box）。
- 无法得到准确的 $g(w)$ 值，只能得到带有噪声的观测值 $\tilde{g}(w, \eta) = g(w) + \eta$ (很合理吧，观测、采样得到的值总是和真实值有差异啊)。


**RM 迭代公式** ：
$$
w_{k+1} = w_k - a_k \tilde{g}(w_k, \eta_k)
$$

其中 $a_k$ 是步长系数。

**收敛条件 (RM Theorem)** ： 为了保证 $w_k$ 概率为 1 地收敛到真实根 $w^*$，步长序列 $\{a_k\}$ 必须满足：

1. $g(w)$（即梯度）是单调递增的，这意味着梯度的导数（即二阶导数/Hessian 矩阵 $\nabla^2 J(w)$）必须大于 0。

   **二阶导数大于 0** 正是 **原函数 $J(w)$ (目标函数 Loss Function) 是凸函数 (Convex)** 的定义。
2. $\sum_{k=1}^\infty a_k = \infty$：步长之和为无穷大。保证无论初始值离根多远，都能走过去。
3. $\sum_{k=1}^\infty a_k^2 < \infty$：步长平方和有限。保证随着时间推移，步长衰减得足够快，使得最终能够消除噪声 $\eta$ 的影响并稳定下来。

#### 3. 为什么 SGD 是 RM 的特例？

在深度学习和 Policy Gradient 中常用的 **随机梯度下降 (SGD)** 其实就是 RM 算法的应用。

- **优化目标**：$\min J(w)$。

- **转化为求根问题**：极值点的梯度为 0，即求解 $\nabla_w J(w) = 0$ 。

- **RM 对应**：

  - $g(w) = \nabla_w J(w)$ (真实的梯度期望)。

  - $\tilde{g}(w) = \nabla_w f(w, x_k)$ (基于单个样本计算出的随机梯度) 。

- **公式**：$w_{k+1} = w_k - \alpha_k \nabla_w f(w, x_k)$。这就是 SGD。

#### 4. 核心：RM 算法在强化学习中的映射

强化学习中几乎所有 Model-free 算法（Q-Learning, TD-Learning, Policy Gradient）的更新公式，本质上都是在求解某个 $g(w)=0$。

**强化学习算法如何对应 RM 框架**的栗子：

| **RL 算法**                      | **待求解变量 w**   | **要逼近的目标 (Root Finding) g(w)=0**                       | **噪声观测 g~ (实际更新量)**                | **对应 RM 公式解释**                                         |
| -------------------------------- | ------------------ | ------------------------------------------------------------ | ------------------------------------------- | ------------------------------------------------------------ |
| **平均值估计** (Mean Estimation) | 估计值 $\hat{\mu}$ | $\hat{\mu} - E[X] = 0$   (估计值应等于期望)                  | $\hat{\mu} - x_k$                           | $w \leftarrow w - \alpha(w - x)$                             |
| **TD Learning** (Value Function) | 价值 $V(s)$        | **Bellman Error = 0**   $V(s) - E[r + \gamma V(s')] = 0$     | **TD Error**   $V(s) - (r + \gamma V(s'))$  | $V(s) \leftarrow V(s) - \alpha \cdot \text{TD\_Error}$   或写成   $V(s) \leftarrow V(s) + \alpha (Target - V(s))$ |
| **Q-Learning**                   | Q值 $Q(s,a)$       | **Bellman Optimality Error = 0**   $Q(s,a) - E[r + \gamma \max Q(s', a')] = 0$ | $Q(s,a) - (r + \gamma \max_{a'} Q(s', a'))$ | $Q \leftarrow Q - \alpha (Q - (r + \gamma \max Q'))$         |
| **Policy Gradient** (REINFORCE)  | 策略参数 $\theta$  | **Gradient Ascent**   $\nabla_\theta J(\theta) = 0$          | $-\nabla_\theta \log \pi(a\mid s) \cdot G_t$    | 负号是因为 RM 是找零点通常对应下降，这里是上升               |

#### 5. 深度理解：TD Learning 中的 RM 视角

TD 算法是特殊的 SA (Stochastic Approximation) 算法 。考虑 **Temporal-Difference (TD)** 更新过程：

$$
V(s_t) \leftarrow V(s_t) + \alpha [ \underbrace{r_{t+1} + \gamma V(s_{t+1})}_{\text{Target}} - V(s_t) ]
$$

如果不看 RM 理论，这看起来只是简单的“向目标靠近一点”。但从 RM 角度看：

1. 我们想解的方程是：**Bellman Equation**。

   $$
   g(V) = E_\pi [r + \gamma V(s')] - V(s) = 0
   $$

2. 但是无法算出期望 $E$（不知道环境模型 $P(s'|s)$）。

3. 所以采样一次转移 $(s_t, a_t, r_{t+1}, s_{t+1})$，得到**含噪的 Bellman Error**：

   $$
   \tilde{g} = (r_{t+1} + \gamma V(s_{t+1})) - V(s_t)
   $$

4. 根据 RM 公式 $w \leftarrow w - a \tilde{g}$ (注意符号方向取决于定义)，用样本的误差来更新参数。

5. **结论**：只要步长 $\alpha$ 满足 RM 条件（逐渐衰减），且策略能遍历所有状态，TD 学习算出的 $V(s)$ **一定**会收敛到 Bellman 方程的真实解。

#### 总结

RM 算法给了强化学习“乱走”的勇气：即使每一步走的方向（随机梯度/TD误差）都不完全对，都包含噪声，但只要大方向是对的（无偏估计），并且步长控制得当（RM 条件），最终一定能走到终点（最优策略/真实价值）。