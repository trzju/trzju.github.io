---
title: 2. 基于模型的方法
date: 2026-1-03
categories: [Reinforcement Learning]
tags: [RL, Notes]
---

### Value iteration

其实在上一章BOE中已经介绍过了

Considering **BOE** and **conraction mapping theorem**, we have 
$$
v_{k+1}=f(v_k)=\max_\pi(r_\pi+\gamma P_\pi v),\ \ \ \ \ \ \ \  k=1,2,3,\cdots
$$
where $v_0$ can be arbitrary. 

- Through this algorithm we can eventually find the optimal state value and the optimal policy.
- This algorithm is called **value iteration**. 

> 在强化学习中，价值迭代的迭代公式 $v_{k+1} = T^*(v_k) = \max_\pi (r_\pi + \gamma P_\pi v_k)$ 的理论根基正是**压缩映射定理（Contraction Mapping Theorem）**。以下从数学原理与算法意义两方面解析：
>
> ---
>
> **一、公式本质：贝尔曼最优算子的显式表达**
>
> - **$T^*$ 的定义**：该公式是贝尔曼最优算子 $T^*$ 的矩阵向量形式。其中：
>   - $r_\pi$：策略 $\pi$ 下的期望即时奖励向量；
>   - $P_\pi$：策略 $\pi$ 对应的状态转移概率矩阵；
>   - $\max_\pi$：**逐状态独立优化**（非全局策略搜索），等价于对每个状态 $s$ 计算 $\max_a \left[ r(s,a) + \gamma \sum_{s'} P(s'|s,a) v_k(s') \right]$，即选择使 Q 值最大的动作（贪婪策略）。
> - **物理意义**：每轮迭代对所有状态执行“全宽度备份”（full-width backup），综合所有可能动作与后继状态，取最优动作对应的估值更新当前状态价值。
>
> ---
>
> **二、压缩映射定理如何保证收敛？**
>
> 1. **度量空间设定**  
>    价值函数空间 $\mathbb{V} = \mathbb{R}^{|S|}$ 配备无穷范数 $\|u - v\|_\infty = \max_{s} |u(s) - v(s)|$，构成**完备度量空间**（知识库[4]）。
>
> 2. **$T^*$ 是 $\gamma$-压缩映射**（关键证明）  
>    对任意 $u, v \in \mathbb{V}$：
>    $$
>    \begin{aligned}
>    \|T^*(u) - T^*(v)\|_\infty 
>    &= \max_s \left| \max_a \mathcal{Q}_u(s,a) - \max_a \mathcal{Q}_v(s,a) \right| \\
>    &\leq \max_s \max_a \left| \mathcal{Q}_u(s,a) - \mathcal{Q}_v(s,a) \right| \quad (\text{利用 } |\max f - \max g| \leq \max |f-g|) \\
>    &= \gamma \max_s \max_a \left| \sum_{s'} P(s'|s,a)(u(s') - v(s')) \right| \\
>    &\leq \gamma \|u - v\|_\infty \quad (\text{因 } \sum_{s'} P(\cdot) = 1)
>    \end{aligned}
>    $$
>    由于 $\gamma \in [0,1)$，$T^*$ 满足压缩映射条件（知识库[4][1]）。
>
> 3. **收敛性结论**（压缩映射定理直接应用）  
>
>    - **唯一不动点**：存在唯一 $v^*$ 使 $T^*(v^*) = v^*$，即满足贝尔曼最优方程的最优价值函数；
>    - **全局收敛**：从任意 $v_0$ 出发，迭代序列 $\{v_k\}$ 收敛至 $v^*$；
>    - **误差界**：$\|v^* - v_k\|_\infty \leq \frac{\gamma^k}{1-\gamma} \|v_1 - v_0\|_\infty$，呈**线性收敛**（知识库[1][4]）。
>
> ---
>
> **三、与策略迭代的关系及算法意义**
>
> - **截断视角**：价值迭代可视为策略迭代的“极端简化”——策略评估（Policy Evaluation）仅执行**单步备份**（而非迭代至收敛），随即进行策略改进（Policy Improvement）（知识库[2][5][7]）。
> - **计算效率**：避免内层策略评估循环，单次迭代成本低，适合中等规模问题（知识库[4]提及“全宽度备份”对百万级状态有效）；
> - **策略隐含性**：迭代中不显式存储策略 $\pi_k$，最优策略 $\pi^*$ 可在收敛后通过 $\pi^*(s) = \arg\max_a [r(s,a) + \gamma \sum_{s'} P(s'|s,a) v^*(s')]$ 提取；
> - **理论桥梁**：压缩映射定理统一解释了价值迭代、策略评估（$T^\pi$ 也是压缩映射）的收敛性（知识库[4]），为动态规划方法提供严格数学基础。
>
> ---
>
> **总结**
>
> 该迭代公式并非经验设计，而是**贝尔曼最优方程在压缩映射框架下的自然迭代解法**。压缩映射定理不仅证明了收敛性与唯一性，还量化了收敛速度，使价值迭代成为动态规划中可靠且高效的求解工具。理解此公式，需把握“算子压缩性→不动点存在→迭代收敛”这一逻辑链，这也是强化学习理论严谨性的核心体现。

#### some details

#### two steps

1. **policy update**: to solve 
   $$
   \pi_{k+1} = \arg \max_\pi (r_\pi + \gamma P_\pi v_k)
   $$
   where $v_k$ is given. 

   即给定 $v_k$，求解策略 $\pi$。

2. **value update**: 
   $$
   v_{k+1}=r_{\pi_{k+1}}+\gamma P_{\pi_{k+1}}v_k
   $$
   带入第一步求解的 $\pi_{k_+1}$, 求解 $v_{k+1}$. ——**问**：为什么这一个迭代式子用上一步迭代的 $v_k$ 来更新 $v_{k+1}$ 可以让 state value 更优？

**注意** 这里的 $v_k$ 是 state value 吗？？

**不是！** 第二步的值迭代式子里，$v_{k+1}$ 和 $v_k$ 就不一定满足 bellman equation 这里 $v_k$ 就是普通的**值向量**。

- 在第一步中的 $v_k$，$v$ 是 state value，但是下标 $k$ 是表示第 k 次迭代。

  也就是说，$v_k$ 代表的是第 k 次迭代的 state value，它不代表任何策略的真实 state value，仅仅是逼近 $v^*$ 的中间迭代量。

- $\pi_{k+1}$ 是确定性策略(由 argmax 生成)，可表示为向量(特指**离散有限状态空间下确定性策略的存储形式**，本质是策略函数 $\pi: S \rightarrow A$ 的离散化实现，不是说有很多个策略组成了 $\pi_{k+1}$ )，就是**一个确定性策略**，向量指的是这个策略的离散数学表达就是对于状态空间的所有状态组成的状态序列，这个策略给到一个对应的动作序列，即在什么状态应该采取什么动作——就是确定性策略，就是动作概率分布的特殊形式，就是策略函数的定义，没什么特殊的。

  其下标 k+1 表示**由第 k 次迭代的 $v_k$ 导出的、用于第 k+1 次价值更新的策略**，纯属算法迭代索引。与马尔可夫过程步骤时间步无关。

- 在第二步中，$v_{k+1}=r_{\pi_{k+1}}+\gamma P_{\pi_{k+1}}v_k$ 是单步贝尔曼备份结果，**非策略 $\pi_{k+1}$ 的真实 state value $v_{k+1}$ **

- 其下标 k 与 k+1 分别代表当前轮次和更新后的迭代轮次。与马尔可夫过程中与环境交互的时间步 $t$ 无关。

**不过呢，contraction mapping theorem and Martrix-vetor form 一般都只是在理论分析中使用，elementwise form 才一般在算法实现中使用。**

#### implementation: elementwise form

1. Policy update
2. Value update

#### **Procedure summary**

$$
v_k(s) \rightarrow q_k(s,a) \rightarrow greedy\ policy\ \pi_{k+1}(a|s) \rightarrow new\ value\ v_{k+1}=\max_aq_k(s,a)
$$

**Initialization**: The probability model (envrioment model) $p(r|s,a), p(s'|s,a)$ for all $(s,a)$ are known. Initial guess $v_0$. 

**Aim**: Search the optimal state value and the optimal policy to solve the bellman optimality equation. 

**Iteration steps**: 

TODO



### Policy iteration

**description**: given a random policy $\pi_0$. 

1. Step 1: Policy evaluation

   对于 elementwise form 与 matrix-vector form，基本都是初始猜测 $v_{\pi_k}$，然后在迭代式子中不断迭代来接近真实值。

2. Step 2: Policy improvement

### Truncated policy iteration  algorithm

截断策略迭代算法介于策略迭代与价值迭代算法中间，或者说策略迭代与价值迭代分别是截断策略迭代算法的两个极端。

见强化学习的数学原理课程slides

这里的价值迭代与策略迭代基于环境模型，如果在算法中实现，多基于动态规划 (Dynamic Programming) 求解，在计算量上对多数问题是不现实的。而且多数强化学习问题也很难对环境进行建模。