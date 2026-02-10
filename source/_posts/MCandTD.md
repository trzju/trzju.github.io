---
title: 3. MC 与 表格TD
date: 2026-1-04
categories: [Reinforcement Learning]
tags: [RL, Notes]
---

## Monte Carlo Learning

上一章 DP（value iteration / policy iteration）是 **model-based**：要知道 $p(s'|s,a), p(r|s,a)$。  
从这章开始转向 **model-free**：只用采样数据（episodes / transitions）估计价值并改进策略。

### MC 的核心逻辑链：how, what, why

- **What**：要估计的是价值函数（尤其是 $q_\pi(s,a)$）。
- **How**：不用模型，直接采样完整 episode，计算 return，做均值估计。
- **Why**：大数定律保证样本均值在样本数足够大时逼近期望值。

### MC 的基本对象与变量（非常关键）

设在时刻 $t$ 的采样轨迹片段为
$$
(S_t,A_t,R_{t+1},S_{t+1},A_{t+1},R_{t+2},\dots)
$$
定义回报（return）：
$$
G_t \triangleq R_{t+1}+\gamma R_{t+2}+\gamma^2R_{t+3}+\cdots
$$

- $S_t$：时刻 $t$ 的状态（state）。
- $A_t$：时刻 $t$ 在状态 $S_t$ 下执行的动作（action）。
- $R_{t+1}$：执行 $A_t$ 后，从环境得到的一步奖励。
- $\gamma\in[0,1)$：折扣因子；越大越重视长期回报。
- $G_t$：从时刻 $t$ 往后累计折扣奖励，是随机变量。

动作价值定义：
$$
q_\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]
$$

MC 的估计思想就是：拿很多个“从 $(s,a)$ 出发并按 $\pi$ 走完”的 episode，取这些 $G_t$ 的平均。

### MC Basic：把 policy iteration 变成 model-free

对于第 $k$ 次迭代，给定策略 $\pi_k$：

1. **Policy evaluation（采样版）**  
   采样估计 $q_{\pi_k}(s,a)$，记估计值为 $q_k(s,a)$。
2. **Policy improvement（贪婪）**  
   $$
   \pi_{k+1}(s)=\arg\max_a q_k(s,a)
   $$

这就是“策略迭代的 model-free 版本”。

### MC Exploring Starts（ES）

Exploring Starts 假设每个状态动作对 $(s,a)$ 都有机会作为 episode 起点被采到。  
这个假设在理论上强（保证覆盖），在工程上通常难严格满足。

常见统计量：

- `Returns(s,a)`：累计所有访问到 $(s,a)$ 时的 return 之和。
- `Num(s,a)`：$(s,a)$ 被用于更新的次数。
- $q(s,a)=\dfrac{\text{Returns}(s,a)}{\text{Num}(s,a)}$：当前 MC 均值估计。

### 提高可实现性：soft policy 与 $\epsilon$-greedy

为了解决“纯贪婪策略探索不足”，用 $\epsilon$-greedy：

设 $a^*(s)=\arg\max_a q(s,a)$，则
$$
\pi(a|s)=
\begin{cases}
1-\epsilon+\epsilon/|\mathcal{A}(s)|,& a=a^*(s)\\
\epsilon/|\mathcal{A}(s)|,& a\neq a^*(s)
\end{cases}
$$
（有些实现把非贪婪项写成 $\epsilon/(|\mathcal{A}(s)|-1)$，本质都是“给非贪婪动作保留正概率”。）

- $\epsilon$ 大：探索强，但策略最优性变差。
- $\epsilon$ 小：更接近贪婪，但探索不足。

结论：**MC + $\epsilon$-greedy = 在“可探索性”和“利用当前估计”之间折中。**

---

## Stochastic approximation

这一章是 MC 到 TD 的数学桥梁：  
“用带噪声的随机迭代，去逼近某个方程/最优点的解”。

### 1) 从均值估计看增量更新

均值估计可写成
$$
w_{k+1}=w_k+\alpha_k(x_k-w_k)
$$

- $w_k$：第 $k$ 次迭代时对目标均值的估计。
- $x_k$：第 $k$ 个样本。
- $(x_k-w_k)$：当前样本与估计的误差（innovation）。
- $\alpha_k$：步长；控制“这次样本对估计影响有多大”。

当 $\alpha_k=1/k$ 时，上式等价于样本均值递推。

### 2) Robbins-Monro（RM）算法

目标：求解根
$$
g(w)=0
$$
但只能观测有噪声的 $\tilde g(w_k,\eta_k)$，于是用
$$
w_{k+1}=w_k-a_k\tilde g(w_k,\eta_k)
$$

- $g(w)$：真实函数（通常未知闭式或难直接计算）。
- $\tilde g(w_k,\eta_k)$：对 $g(w_k)$ 的随机观测（unbiased/noisy measurement）。
- $\eta_k$：噪声随机变量。
- $a_k$：步长序列。

典型收敛步长条件：
$$
\sum_{k=1}^\infty a_k=\infty,\qquad \sum_{k=1}^\infty a_k^2<\infty
$$
例如 $a_k=\frac{1}{k}$。

### 3) SGD 是 RM 的特例

若要最小化
$$
J(w)=\mathbb{E}[f(w,X)]
$$
则根方程是
$$
g(w)=\nabla_wJ(w)=0
$$
SGD 更新：
$$
w_{k+1}=w_k-a_k\nabla_w f(w_k,x_k)
$$
本质上就是把 $\nabla_wJ(w)$ 用单样本随机梯度近似。

---

## Temporal-Difference Learning

和 MC 一样是 model-free，但核心区别：

- MC：等 episode 结束后用完整 return，**non-incremental / full return**。
- TD：每步就更新，bootstrap，**incremental / one-step bootstrapping**。

### TD(0) 状态价值学习

给定策略 $\pi$，目标是估计 $v_\pi(s)$。  
TD(0) 更新：
$$
v_{t+1}(s_t)=v_t(s_t)+\alpha_t(s_t)\delta_t
$$
其中
$$
\delta_t\triangleq r_{t+1}+\gamma v_t(s_{t+1})-v_t(s_t)
$$

- $v_t(s_t)$：更新前对当前状态价值的估计。
- $r_{t+1}+\gamma v_t(s_{t+1})$：TD target（一步 bootstrap 目标）。
- $\delta_t$：TD error（目标与当前估计的差）。
- 其他状态 $s\neq s_t$ 的值不变（只局部更新当前访问状态）。

### Sarsa（on-policy action-value TD）

$$
q_{t+1}(s_t,a_t)=q_t(s_t,a_t)+\alpha_t(s_t,a_t)\Big[r_{t+1}+\gamma q_t(s_{t+1},a_{t+1})-q_t(s_t,a_t)\Big]
$$

- TD target 用的是 **同一行为策略**在下一状态实际选出的 $a_{t+1}$。
- 所以是 on-policy。

### n-step Sarsa：统一 Sarsa 和 MC

$$
G_t^{(n)}=r_{t+1}+\gamma r_{t+2}+\cdots+\gamma^{n-1}r_{t+n}+\gamma^n q_t(s_{t+n},a_{t+n})
$$
然后
$$
q_{t+1}(s_t,a_t)=q_t(s_t,a_t)+\alpha_t\big[G_t^{(n)}-q_t(s_t,a_t)\big]
$$

- $n=1$：退化为 one-step Sarsa。
- $n\to\infty$（终止前）：趋近 MC return。

### Q-learning（off-policy，直接逼近最优）

$$
q_{t+1}(s_t,a_t)=q_t(s_t,a_t)+\alpha_t(s_t,a_t)\Big[r_{t+1}+\gamma\max_a q_t(s_{t+1},a)-q_t(s_t,a_t)\Big]
$$

- TD target 里是 $\max_a$，对应最优贝尔曼方程。
- 采样动作来自 behavior policy（可探索），但学习目标是 greedy target policy。
- 因此是 off-policy。

### TD 系列统一视角

基本都可写成
$$
\text{Estimate}_{new}=\text{Estimate}_{old}+\alpha(\text{Target}-\text{Estimate}_{old})
$$
差异都在 **Target 的定义**（MC return / bootstrap / max / n-step）。
