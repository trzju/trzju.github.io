---
title: 1. 强化学习基本工具
date: 2026-1-02
categories: [Reinforcement Learning]
tags: [RL, Notes]
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
