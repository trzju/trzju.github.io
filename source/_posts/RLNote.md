---
title: 强化学习的数学原理 | CS 285
date: 2025-12-21
categories: [Reinforcement Learning]
tags: [RL, Math, Notes]
---


RL 的各种**分类方法**: 

- Model-based, Model-free. 

  > Most questions in RL are Model-free. 

- Policy-based, Value-based, Actor-Critic. 

- 回合更新, 单步更新. Non-incremental, incremental. 

  > MC, TD. 

- on policy, off policy. 

- online, offline

RL 任务的**分类方法**: 

- 离散型空间与连续型空间

  > continuous space and discrete space. 

- 回合制任务与持续性任务

  > episodeic, continuing. 

## starting overview

强化学习的最终目标：**最优策略 optimal policy**. 

考虑课程 CS285 常用的以下式子：
$$
\underbrace{p_\theta (s_1,a_1,\cdots,s_T,a_T)}_{p_\theta (\tau)} = p(s_1)\prod^T_{t=1}\underbrace{ \pi_\theta (a_t|s_t)p(s_{t+1}|s_t,a_t)}_{\text{Markov Chain on }(s,a)}
$$
其中，$s_t,a_t$ 均为随机变量，$\tau$ 代表轨迹 (trajectory) 这一随机变量，在此处以 $s_1, a_1, \cdots, s_T, a_T$ 的有限状态动作对数据序列表示出来。$s_1$ 是初始状态。每个轨迹的观测值都由策略与环境交互得来。当然对于部分观测马尔可夫过程(POMDP)来说此处的 $s$ 应该替换为 $o$ 。

$\theta$ 代表策略 $\pi(a|s)$ 的参数，当我们以一个神经网络来拟合策略函数时，$\theta$ 就是该神经网络的权重参数。

$p_\theta (\tau)$ 就是轨迹的对应观测值的概率，即轨迹的分布。下标 $\theta$ 表示该概率分布依赖于策略的参数 $\theta$，即我们关注该参数，目标就是优化参数 $\theta$ 使得轨迹这一随机变量的回报(奖励 $r_t$ 的累积)的加权期望最优。

即：
$$
\theta^* = \arg \max_\theta E_{\tau \sim p_\theta(\tau)}[\sum_t r(s_t,a_t)]
$$
考虑 $p_\theta (s,a)$ 为 state-action 的边缘分布 (此处 $p_\theta(s,a)$ 非平稳分布 stationary distribution)，

finite horizon case: 
$$
\theta^*=\arg \max_\theta \sum^T_{t=1}E_{(s_t,a_t)\sim p_\theta(s_t,a_t)}[r(s_t,a_t)]
$$
infinite horizon case: 
$$
\theta^*=\arg \max_\theta E_{(s,a)\sim p_\theta(s,a)}[r(s,a)]
$$
考虑到实际算法的收敛性，通常会引入折扣因子 (或者在计算时截断trajectory) ，即常见的优化目标是折扣回报期望：
$$
J(\theta) = E_{\tau \sim p_\theta(\tau)} [\sum_{t=1}^\infty \gamma^{t-1} r(s_t, a_t)]
$$


在强化学习中，我们总是关注**期望**。

> 对于离散动作空间来说，为什么动作与奖励都不是平滑的，但是却可以用梯度下降等平滑方法来优化呢？
>
> 因为即使动作与奖励都不是平滑的，**奖励关于策略的期望却是关于策略的参数 $\theta$ 平滑变化的**。

在强化学习算法中，几乎总是分为以下三个循环步骤：

1. generate samples (i.e. run the policy) 
2. fit a model / estimate the return 
3. improve the policy 

三个步骤循环进行。



定义Q函数为从状态 $s_t$ 开始采取 $a_t$ 后获得的总奖励的期望：
$$
\begin{align} Q^\pi(s_t,a_t)=& \sum^T_{t'=t}E_{\pi_\theta}[r(s_{t'},a_{t'})|s_t,a_t] \\ =& E_{\tau \sim p_\theta(\tau)} \left[ \sum_{t'=t}^T r(s_{t'}, a_{t'}) \mid s_t, a_t \right] \end{align}
$$
定义价值函数为从状态 $s_t$ 开始获得的总奖励：
$$
\begin{align} V^\pi(s_t)=& \sum^T_{t'=t}E_{\pi_\theta}[r(s_{t'},a_{t'})|s_t] \\ =& E_{a_t\sim\pi(a_t|s_t)}[Q^\pi(s_t,a_t)] \end{align}
$$
强化学习问题的**首要目标**则是：
$$
E_{s_1\sim p(s_1)}[V^\pi(s_1)]
$$
强化学习中两个重要的基础想法：

- 如果我们有策略 $\pi$，而且知道 $Q^\pi(s,a)$，那么就可以改进策略 $\pi$ ：

  set $\pi'(a|s)=1$ if $a=\arg \max_aQ^\pi(s,a)$ （贪婪化）

  this policy is at least as good as $\pi$ , and it doesn't matter waht $\pi$ is. 

- 通过计算梯度来增加好动作 $a$ 的概率 (涉及到最大似然估计与加权最大似然估计) 

  if $Q^\pi(s,a)>V^\pi(s)$ , then $a$ is better than average ( $V^\pi(s)=E[Q^\pi(s,a)]$ under $\pi(a|s)$ , 从定义即可很好理解，也符合直觉 )

  modify $\pi(a|s)$ to increase probillity of $a$ if $Q^\pi(s,a)>V^\pi(s)$ . ---->> 引入优势函数 $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$ 。





# Math Foundation of Reinforcement Learning

## **Bellman equation and derivation **

$$
\begin{align} v_\pi(s) =& \mathbb{E}[G_t|S_t=s] \\ =& \mathbb{E}[R_{t+1}+\gamma G_{t+1}|S_t=s] \\ =& \mathbb{E}[R_{t+1}|S_t=s] + \gamma\mathbb{E}[G_{t+1}|S_t=s] \\ =& \sum_a \pi(a|s) \mathbb{E}[R_{t+1}|S_t=s,A_t=a] + \gamma \sum_{s'} \mathbb{E}[G_{t+1}|S_t=s,S_{t+1}=s']p(s'|s) \\ =& \sum_a\pi(a|s)\mathbb{E}[R_{t+1}|S_t=s,A_t=a]+\gamma \sum_{s'} v_\pi(s')p(s'|s) \\ =& \sum_a \pi(a|s) \sum_rp(r|s,a)r +\gamma \sum_{s'}v_\pi(s')\sum_a\pi(a|s)p(s'|s,a) \\ =& \sum_a\pi(a|s)[\sum_rp(r|s,a)r+\gamma \sum_{s'}p(s'|s,a)v_\pi(s')] \end{align}
$$

**let**
$$
v_\pi(s)=r_\pi(s)+\gamma \sum_{s'}p_\pi(s'|s)v_\pi(s')
$$
**where**
$$
r_\pi(s) \triangleq \sum_a\pi(a|s)\sum_rp(r|s,a)r \\ p_\pi(s'|s) \triangleq \sum_a\pi(a|s)p(s'|s,a)
$$

> $\triangleq$ ---- is defined as 定义为
>
> 人为的符号约定
>
> “**定义**策略 $\pi$ 诱导的状态转移概率 $p_\pi(s'|s)$ 为：
>
> 对任意状态 $s, s' \in \mathcal{S}$ ，$p_\pi(s' \mid s) \triangleq \sum_{a \in \mathcal{A}} \pi(a \mid s) p(s' \mid s, a),$其中 AA 为动作空间。
>
> 该定义表示：在策略 $\pi$ 下，从状态 $s$ 转移到 $s'$ 的概率，是通过对所有可能动作 $a$ 应用全概率公式进行边缘化得到的加权和。”

**for states $s_i$ ($s_i = 1, \cdots, n$), the bellman eqution is** 
$$
v_\pi(s_i)=r_\pi(s_i)+\gamma \sum_{s_j}p_\pi(s_j|s_i)v_\pi(s_j)
$$
**put all these equations for all the states together and rewrite to a matrix-vector form**
$$
v_\pi=r_\pi+\gamma P_\pi v_\pi
$$
**where**

- $v_\pi=[v_\pi(s_1),\cdots,v_\pi(s_n)]^T\in\mathbb{R}^n$
- $r_\pi=[r_\pi(s_1),\cdots,r_\pi(s_n)]^T\in\mathbb{R}^n$
- $P_\pi\in\mathbb{R}^{n\times n}$, where $[P_\pi]_{ij}=p_\pi(s_j|s_i)$, is **the state transition matrix**.

> **? WHY** $\sum_a\pi(a|s)\sum_{s'}p(s'|s,a)v_\pi(s')=\sum_{s'}\sum_a\pi(a|s)p(s'|s,a)v_\pi(s')\triangleq \sum_{s'}p_\pi(s'|s)v_\pi(s')$ 在第一个等号变换中只是交换了求和顺序
>
> 在内层求和 $\sum_{s'}$ 中，$\pi(a|s)$ 是**与 $s'$ 无关的常数**（因为给定状态 $s$，策略 $\pi(a|s)$ 只依赖于动作 $a$，不依赖于下一个状态 $s'$）。因此，我们可以把整个表达式看作对所有 $(a, s')$ 对的联合求和：$= \sum_a \sum_{s'} \pi(a|s) \, p(s'|s,a) \, v_\pi(s') $
> 这一步只是将外层的 $\pi(a|s)$ 写进内层求和中——这是合法的，因为它是关于 $s'$ 的常数。
>
> 然后 $p_\pi(s'|s) \triangleq \sum_a\pi(a|s)p(s'|s,a)$ : 
>
> - 这个式子中 $p_\pi(s'|s)$ 表示策略 $\pi$ 下从给定状态 $s$ 转移到状态 $s'$ 的**概率分布**，这个过程在**马尔可夫过程**中本来就涉及到agent采取action $a$ 的概率分布，因此可以展开为后面的式子也是符合直觉的——**见下一个引用块**。
>
> - 强化学习中**状态转移概率在策略下的边缘化（marginalization）** 的核心思想
>
> - 这是**在策略 $\pi$ 下，从状态 $s$ 转移到状态 $s'$ 的边缘概率**（也称为“策略诱导的状态转移核”）
>
> - **当我们只关心状态到状态的转移（而不关心中间采取了什么动作）时，需要对所有可能的动作 $a$ 进行加权平均，权重就是策略 $\pi(a|s)$。**
>
> - 在贝尔曼期望方程中：
>   $$
>   v_\pi(s) = \sum_a \pi(a|s) \left[ \sum_r p(r|s,a) r + \gamma \sum_{s'} p(s'|s,a) v_\pi(s') \right]
>   $$
>
>   我们可以将后半部分重写为：
>   $$
>   \gamma \sum_{s'} \left( \sum_a \pi(a|s) p(s'|s,a) \right) v_\pi(s') = \gamma \sum_{s'} p_\pi(s'|s) v_\pi(s')
>   $$
>
>   这就把**动作显式的表达式**转化为了**仅关于状态的马尔可夫链形式**，说明在策略 $\pi$ 下，状态序列 $\{s_t\}$ 本身构成一个**马尔可夫链**，其转移核就是 $p_\pi(s'|s)$。
>
> - 这一操作使我们能将带动作的 MDP 抽象为策略 $\pi$ 诱导的**状态马尔可夫链**，是理论分析的重要工具
>
> - 这是一个标准的**全概率公式（Law of Total Probability）** 的应用

> 每次将均值 $\mathbb{E}[]$ 展开为 $\sum$，就是对某一个变量乘以这个变量的概率分布求和(理解均值的意义)。
>
> 关键点1：$\mathbb{E}[G_{t+1}|S_t=s,S_{t+1}=s']=\mathbb{E}[G_{t+1}|S_{t+1}=s']=v_\pi(s')$ —— 马尔可夫过程的无记忆性，以及状态价值函数的定义。
>
> 关键点2：$\mathbb{E}[R_{t+1}|S_t=s,A_t=a]=\sum_rr\cdot p(r|s,a)$ —— 采取行动获得奖励的**随机性**
>
> 关键点3：$\sum$ 求和符号可以像乘法因子一样满足交换律吗？---- 这不是乘法交换，是线性运算的代数规则：双重求和在满足绝对收敛等条件下可以**交换求和顺序**；若因子与求和指标无关，可提取至求和外。

OK，所以可以看出**贝尔曼方程就是不同状态的状态价值函数之间的关系**，该式子对所有的 $s \in \mathcal{S}$ 都成立，即对于一个马尔可夫过程，有 $n$ 个状态 $s$ 就有 $n$ 个贝尔曼方程成立 (对于最终状态 $s$ 那么对应的贝尔曼方程的两个价值函数就都是 $v_\pi(s)$ )。

在式子中，$\pi(a|s)$ 是给定(given)的，因此贝尔曼方程可以用于评估(evaluate)、改进**策略**。

用矩阵向量形式的贝尔曼方程可以求解状态价值

Why to solve state values?

- Given a policy, finding out the corresponding state values is called **policy evaluation**!
- It is fundamental problem in RL. It is the fundation to find better policies.
- Therefore, it is important to understand how to solve the bellman equation.

**如何求解？理论证明、求闭合解、数值迭代方法等**

**不动点迭代法**

// TODO 是否一定收敛、最优解性质

### more frequently used: action value

- State value: the average return the agent can get *starting from* a state. ---- $v_\pi(s)=\mathbb{E}[G_t|S_t=s]$
- Action value: the average return the agent can geti *starting from* a state and *taking* an action. ---- $q_\pi(s,a)=\mathbb{E}[G_t|S_t=s,A_t=a]$ 

Why do we care action value? Because we want to know which action is better.

- Action-value is the function of the state-action pair (s,a), it **depends on** policy $\pi$. 

有定义可知状态价值函数与动作价值函数的关系:
$$
v_\pi(s)=\sum_a\pi(a|s)q_\pi(s,a)
$$
联系前面推导的贝尔曼方程$v_\pi(s)=\sum_a\pi(a|s)[\sum_rp(r|s,a)r+\gamma \sum_{s'}p(s'|s,a)v_\pi(s')]$, 得出:
$$
q_\pi(s,a)=\sum_rp(r|s,a)r+\gamma\sum_{s'}p(s'|s,a)v_\pi(s')
$$
上面这两个式子即揭示了如何在**所有state**的**状态价值函数**与**动作价值函数**进行转换。

#### optimal policy

Definition: A policy $\pi^*$ is **optimal** if $v_{\pi^*}(s)\ge v_\pi(s)$ for all $s\in \mathcal{S}$ and any other policy $\pi$. 

- Does the optimal policy exist?
- Is the optimal policy unique?
- Is the optimal policy stochastic or deterministic?
- How to obtain the optimal policy?

**SO INTRODUCE THE ------ *Bellman optimality equation.***

## Bellman Optimality Equation

Core concepts in RL: **optimal state value** and **optimal policy**. 

So the **fundamental tool**: the **Bellman Optimality equation**. 

#### **BOE** (elementwise form) 

(对每个向量或元素独立操作的形式)--其实就是从BOE的定义出发得来的式子: 
$$
\begin{align} v(s)=&\max_\pi \sum_a \pi(a|s)(\sum_r p(r|s,a)r+\gamma \sum_{s'}p(s'|s,a)v(s')),s\in \mathcal{S} \\ =&\max_\pi \sum_a \pi(a|s)q(s,a) \end{align}
$$

- $p(r|s,a),p(s'|s,a),r,\gamma$ are **known**.
- $v(s), v(s')$ are **unkonw** and **to be calculated**. 
- $\pi(a|s)$ are unknown. 

#### **BOE** (matrix-vector form) 

$$
v=\max_\pi (r_\pi+\gamma P_\pi v)
$$

where the elements corresponding to $s$ or $s'$ are: 
$$
[r_\pi]_s \triangleq \sum_a \pi(a|s) \sum_r p(r|s,a)r, \\ [P_\pi]_{s',s} = p(s'|s) \triangleq \sum_a \pi(a|s)\sum_{s'} p(s'|s,a)
$$

对于矩阵向量形式的BOE: 

- $v\in \mathbb{R}^{|s|}$ 是待求解的最优价值**向量**，$[v]_s=v(s)$. 

- 左边的 $v$ 是解向量 $v^*$, (所有状态的最优价值)

- 右边的 $v$ 是同一解向量 $v^*$ 作为贝尔曼最优算子 $f(\cdot)$ 的输入

- **本质**：这是一个**不动点方程** $v^*=f(v^*)$, 要求输入 $v^*$ 经算子作用后输出仍为 $v^*$. 

  解向量 $v^*$ 在方程中作为整体出现，无需区分“当前状态”与“下一状态”的符号，**转移依赖已隐含在 $P_\pi v$ 中**。 

- 即使在 elementwise form 的 BOE 中：

  - **$v(s)$ 与 $v(s')$ 并非不同变量**！它们都是**同一最优价值函数 $v^*$ 的分量**：  
    - 左边 $v(s)$：状态 $s$ 的最优价值（待求解）  
    - 右边 $v(s')$：状态 $s'$ 的最优价值（同样是待求解的未知量）  
  - **为何显式写 $v(s')$**？  
    为清晰表达**状态转移的依赖关系**：当前状态 $s$ 的价值依赖于所有可能后继状态 $s'$ 的价值（通过转移概率 $p(s'|s,a)$ 加权）。  
    → 这是**方程组的耦合结构**：对每个 $s$ 写一个方程，所有方程联立求解整个向量 $v^*$。

- here $\max_\pi$ is performed elementwise (逐元素执行): 

$$
\max_\pi \left[ \begin{matrix} * \\ \vdots \\ * \end{matrix} \right] = \left[ \begin{matrix} \max_{\pi(s_1)}* \\ \vdots \\ \max_{\pi(s_1)}* \end{matrix} \right]
$$

- **BOE** describes the *optimal policy* and *optimal state* value in an elegant way. 
- There is an maximization on the right-hand side, which may not be straightforward to see how to compute. 

**NEXT**:

- **Algorithm**: how to solve the equation? 
- **Existence**: does this equation have solutions?
- **Uniqueness**: is the solution to this equation unique?
- **Optimality**: how is it related to optimal policy?

### BOE: Preliminaries

#### How to solve equation with maximization? 

> examples in slides $\cdots$ 

Fix $v'(s)$ and solve $\pi$ : 
$$
\begin{align} v(s) =& \max_\pi \sum_a \pi(a|s) q(s,a), \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ \ s\in \mathcal{S} \\ =& \max_\pi [\pi(a_1|s)q(s,a_1) + \cdots + \pi(a_5|s)q(s,a_5)] \\ =& \max_{c_1,\cdots,c_5} [c_1q(s,a_1)+\cdots+c_5q(s,a_5)], \ \ \  c_1+\cdots+c_5=1 \end{align}
$$
Here, $q(s,a_i)$ are **unkown**, because $v(s')$ are **unkonwn**. 

> Example:
>
> Suppose $q_i$ are given, find $c_i^*$ solving $\max_{c_1,c_2,c_3}c_1q_1+c_2q_2+c_3q_3$, wherer $c_1+c_2+c_3=1$, and $c_1,c_2,c_3 \ge 0$. 
>
> **Answer**: suppose $q_3 \ge q_1,q_2$, then, the optimal solution is $c_3^*=1, c_1^*=c_2^*=0$,  $\max_{c_1,c_2,c_3}c_1q_1+c_2q_2+c_3q_3=q_3$. 
>
> **Why**?
>
> when $q_3 \ge q_1,q_2$, for any case: $q_3=(c_1+c_2+c_3)q_3=c_1q_3+c_2q_3+c_3q_3\ge c_1q_1+c_2q_2+c_3q_3$. 

Therefore: 

considering $\sum_a \pi(a|s)=1$, we have 
$$
v(s)=\max_a \sum_a \pi(a|s) q(s,a) = \max_{a\in \mathcal{A}(s)} q(s,a)
$$
where the optimality is achieved when 
$$
\pi(a|s)= \left\{ \begin{align} 1 \ \  a=a* \\ 0 \ \  a \neq a^* \end{align} \right.
$$
**where** $a^*=\arg \max_a q(s,a)$. 

#### rewrite as $v=f(v)$

**BOE** (matrix-vector form): $v=\max_\pi(r_\pi+\gamma P_\pi v)$. **Let** 
$$
f(v)=\max_\pi (r_\pi + \gamma P_\pi v)
$$
we have 
$$
v=f(v)
$$
where
$$
[f(v)]_s=\max_\pi \sum_a \pi(a|s) q(s,a), \ \ \ \ \ s\in \mathcal{S}
$$

> 1. **$[f(v)]_s = \max_\pi \sum_a \pi(a|s) q(s,a)$ 是 BOE 的 elementwise（逐状态）形式**  
>
>    - 其中 $q(s,a) = \mathbb{E}[r + \gamma v(s') \mid s,a] = \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a)v(s')$，**显式依赖当前 $v$**。  
>
>    - 由于 $\max_\pi \sum_a \pi(a|s) q(s,a) = \max_a q(s,a)$（最优策略在 $s$ 处必为确定性策略），该式等价于标准 BOE：  
>     $$
>      [f(v)]_s = \max_a \left[ \sum_r p(r|s,a)r + \gamma \sum_{s'} p(s'|s,a)v(s') \right]
>     $$
>      **这正是针对单个状态 $s$ 的贝尔曼最优更新规则**。
>
> 2. **$v = f(v) = \max_\pi (r_\pi + \gamma P_\pi v)$ 是 BOE 的 matrix-vector（向量）紧凑形式**  
>
>    - $r_\pi \in \mathbb{R}^{|\mathcal{S}|}$：策略 $\pi$ 下的期望即时奖励向量，$[r_\pi]_s = \sum_a \pi(a|s) \sum_r p(r|s,a)r$  
> 
>    - $P_\pi \in \mathbb{R}^{|\mathcal{S}| \times |\mathcal{S}|}$：策略 $\pi$ 的状态转移矩阵，$[P_\pi]_{s,s'} = \sum_a \pi(a|s) p(s'|s,a)$  
> 
>    - **关键性质**：$\max_\pi$ 操作可**按状态分解**（因策略在不同状态的选择相互独立）：  
>     $$
>      \big[\max_\pi (r_\pi + \gamma P_\pi v)\big]_s = \max_{\pi(\cdot|s)} \big[ r_\pi(s) + \gamma (P_\pi v)(s) \big] = \max_a q(s,a)
>     $$
>      因此，**取该向量方程的第 $s$ 个分量，直接得到 elementwise 形式**。
>
> - **$\max_\pi$ 不是“对整个策略向量取最大”**：  
>  因策略在不同状态的选择独立，$\max_\pi$ 实际等价于 **对每个状态 $s$ 独立取 $\max_a$**（即 $\max_\pi \to \prod_s \max_{\pi(\cdot|s)}$）。这是两种形式能等价转换的数学基础（见 Puterman《MDP》定理 6.2.1）。
> - **$q(s,a)$ 的依赖关系**：  
>   式中 $q(s,a)$ **必须理解为基于当前输入 $v$ 计算的中间量**（即 $q_v(s,a)$），而非固定函数。这是迭代算法（如值迭代）中 $f$ 作为算子的核心。
> 
> - **$[f(v)]_s$ 就是 BOE 的 elementwise 形式**：它显式描述了单个状态 $s$ 的最优价值更新规则。  
> - **Matrix-vector 形式的 BOE 按状态 $s$ 展开，必然得到 elementwise 形式**：二者是同一数学对象（贝尔曼最优算子 $f$）在不同表示粒度下的体现，**无任何矛盾或信息损失**。  
>- 这种“整体向量方程 ↔ 逐状态分量方程”的对应关系，是理解动态规划算法（值迭代/策略迭代）理论基础的关键——算法迭代的是向量 $v$，但每一步更新本质是逐状态应用 elementwise 规则。
> 

How to solve it? 

#### Contraction mapping theorem

// TODO 

### BOE: Solution

Applying the **contraction mapping theorem** gives the following results. 

**For the BOE $v=f(v)=\max_\pi(r_\pi+\gamma P_\pi v)$, there always *exists* a solution $v^*$ and the solution is *unique*. The solution could be solved iteratively by **
$$
v_{k+1}=f(v_k)=\max_\pi(r_\pi+\gamma P_\pi v)
$$
**This sequence $\{v_k\}$ converges to $v^*$ exponentially fast given any initial guess $v_0$. The convergence rate is determined by $\gamma$.   **

**NOTED**: the algorithm in $v_{k+1}=f(v_k)=\max_\pi(r_\pi+\gamma P_\pi v)$ is called the **value iteration algorithm**. 

### BOE: Optimality

OK. Now suppose $v^*$ is the **solution** of the Bellman optimality equation. It satifies
$$
v^* = \max_\pi (r_\pi + \gamma P_\pi v^*)
$$

Then, we can suppose: 
$$
\pi^* = \arg \max_\pi (r_\pi + \gamma P_\pi v^*)
$$
Then: 
$$
v^* = r_{\pi^*}+\gamma P_{\pi^*}v^*
$$
Therefore, $\pi^*$ is a policy and $v^*=v_{\pi^*}$ is the corresponding state value. 

**可以证明** $\pi^*$ 是最优策略，$v^*$ 是最优价值函数。即 $v^* \ge v_\pi , \  \forall \pi$

即BOE方程的解就是**最优价值函数**，也可以由此得到**最优策略**。

最优策略长什么样？
$$
\pi^*(s) = \arg \max_\pi \sum_a \pi(a|s)q^*(s,a)
$$
For any $s \in \mathcal{S}$, the **deterministic greedy** policy 
$$
\pi^*(s) = \left\{ \begin{align} \  1\ \ \ \ a=a^*(s) \\ \ 0 \ \ \ \ a\neq a^*(s) \end{align} \right.
$$
is an **optimal policy** solving the **BOE**. Here, 
$$
a^*(s) = \arg \max_a q^*(a,s),
$$
也就是说，**最优策略**是**决定的、贪婪的策略**，是BOE的解

### BOE: analyzing optimal policy

什么因素**决定了**最优策略、最优价值函数？？

从逐元素形式的BOE以及其中的已知量、未知量易知：

三个因素决定 optimal policy and optimal value

- Reward design: $r$ 
- System model: $p(s'|s,a), p(r|s,a)$ 
- Discount rate: $\gamma$ 
- And $v(s),v(s'),\pi(a|s)$ are unkown values to be calculated

Reward $\mathrm{r} \rightarrow a\cdot \mathrm{r} +b$, this won't change the optimal policy! 

### what ahout all equations above in continous space

## Value iteration and Policy iteration

In this chapter, all these two algorithm are **model-based**. —— 即环境模型的几个概率分布是已知的。

Three sections (value iteration, policy iteration, truncated policy iteration) below are tightly connected. 

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
> ### 一、公式本质：贝尔曼最优算子的显式表达
>
> - **$T^*$ 的定义**：该公式是贝尔曼最优算子 $T^*$ 的矩阵向量形式。其中：
>   - $r_\pi$：策略 $\pi$ 下的期望即时奖励向量；
>   - $P_\pi$：策略 $\pi$ 对应的状态转移概率矩阵；
>   - $\max_\pi$：**逐状态独立优化**（非全局策略搜索），等价于对每个状态 $s$ 计算 $\max_a \left[ r(s,a) + \gamma \sum_{s'} P(s'|s,a) v_k(s') \right]$，即选择使 Q 值最大的动作（贪婪策略）。
> - **物理意义**：每轮迭代对所有状态执行“全宽度备份”（full-width backup），综合所有可能动作与后继状态，取最优动作对应的估值更新当前状态价值。
>
> ---
>
> ### 二、压缩映射定理如何保证收敛？
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
> ### 三、与策略迭代的关系及算法意义
>
> - **截断视角**：价值迭代可视为策略迭代的“极端简化”——策略评估（Policy Evaluation）仅执行**单步备份**（而非迭代至收敛），随即进行策略改进（Policy Improvement）（知识库[2][5][7]）。
> - **计算效率**：避免内层策略评估循环，单次迭代成本低，适合中等规模问题（知识库[4]提及“全宽度备份”对百万级状态有效）；
> - **策略隐含性**：迭代中不显式存储策略 $\pi_k$，最优策略 $\pi^*$ 可在收敛后通过 $\pi^*(s) = \arg\max_a [r(s,a) + \gamma \sum_{s'} P(s'|s,a) v^*(s')]$ 提取；
> - **理论桥梁**：压缩映射定理统一解释了价值迭代、策略评估（$T^\pi$ 也是压缩映射）的收敛性（知识库[4]），为动态规划方法提供严格数学基础。
>
> ---
>
> ### 总结
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



