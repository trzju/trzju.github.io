---
title: 0. 强化学习基本概念
date: 2025-11-30
categories: [Reinforcement Learning]
tags: [RL, Notes]
---

## basic terminology

### 常用符号

| symbol            | zh               | en                            |
| ----------------- | ---------------- | ----------------------------- |
| $S, s$            | 状态             | state                         |
| $A, a$            | 动作             | action                        |
| $R, r$            | 奖励             | reward                        |
| $U, u$            | 回报             | return                        |
| $\gamma$          | 折扣率           | discount factor               |
| $\mathcal{S}$     | 状态空间         | state space                   |
| $\mathcal{A}$     | 动作空间         | action space                  |
| $\pi(a\mid s)$        | 随机策略函数     | stochastic policy function    |
| $\mu(s)$          | 确定策略函数     | deterministic policy function |
| $p(s'\mid s,a)$       | 状态转移函数     | state-transition function     |
| $Q_\pi(s,a)$      | 动作价值函数     | action-value function         |
| $Q_*(s,a)$        | 最优动作价值函数 | optimal action-value function |
| $V_\pi(s)$        | 状态价值函数     | state-value function          |
| $V_*(s)$          | 最优状态价值函数 | optimal state-value function  |
| $D_\pi(s)$        | 优势函数         | advantage function            |
| $D_*(s)$          | 最优优势函数     | optimal advantage function    |
| $\pi(a\mid s;\theta)$ | 随机策略网络     | stochastic policy network     |
| $\mu(s;\theta)$   | 确定策略网络     | deterministic policy network  |
| $Q(s,a;w)$        | 深度Q网络        | deep Q network (DQN)          |
| $q(s,a;w)$        | 价值网络         | value network                 |

### **lecture note**

action, state, reward, return这四个量在RL中都是randomness有随机性的，用大写字母表述随机**变量**，用小写字母表示这个随机**变量**已观测到的量。

**随机策略函数**：

是一个概率密度/质量函数(PDF, PMF)

$\pi(a|s)$ is the probability of taking action $A=a$ given state $s$, e.g., 

- $\pi(left|s)=0.2$
- $\pi(right|s)=0.1$
- $\pi(up|s)=0.7$

（以Mario游戏为例）这表示在当前确定状态$s$下，Mario采取行动$A$（或者$a$？）的概率分布，$A$（或者$a$）可能为向上、向右或者向左，随机策略函数则给出了可能采取的对应行动及其概率。
$$
\pi(a|s)=\mathbb{P}(A=a|S=s)
$$

> 在博弈中，通常都需要采取随机策略而非确定策略，否则对手就可以根据状态预测agent行动

以此来理解**状态转移函数**，也就是在给定状态$s$时，agent采取行动$a$（小写的，一个确定的action，就像给定的状态$s$一样）后环境的下一个状态的可能取值，也是一个PDF(or PMF)，即状态转移函数给出这样的条件下环境的下一个状态的可能取值及对应概率。
$$
p(s'|s,a)=\mathbb{P}(S'=s'|S=s,A=a)
$$
OK, randomness in action and randomness in states基本就是RL中的主要的两个随机性的来源

**奖励与汇报 rewards and returns**

Return -- cumulative future reward
$$
U_t=R_t+R_{t+1}+R_{t+2}+R_{t+3}+\cdots
$$
其实更常见的定义是从 $t+1$ 时刻开始累积。此处来自 Wang Shusen 的课程。

不过不论是 $R_t$ 还是 $R_{t+1}$ 作为首项，都表达的是同一个变量，即 $(S_t,A_t)$ 经过环境模型后得到的 $(R,S_{t+1})$ 中的 $R$ 。

奖励 $R$ 从 $t$ 时刻开始累积，与 $t$ 时刻之前无关

> - 从马尔可夫性质来看，$U_t$ 的**期望值** (即价值函数 $V(s_t)$——btw价值函数才通常是RL中关注的东西之一) 只依赖于当前状态 $s_t$，与 $t$ 时刻之前的动作、状态无关
> - 这是RL的基础：在给定当前状态下，未来奖励的期望只取决于当前状态，而与历史无关
> - $U_t$ 本身是一个随机变量，它的**实际值**会依赖于从 $t$ 时刻开始的未来动作和状态序列
> - 这些未来动作和状态序列是通过策略 $\pi$ 和环境动态决定的，但这些决定只依赖于当前状态 $s_t$，而不是历史

一般来说，还要加上折扣率 $\gamma$ (人为添加，是微调超参数) :
$$
U_t=R_t+\gamma R_{t+1}+\gamma ^2 R_{t+2}+\gamma ^3 R_{t+3}+\cdots
$$

- Discounted return (at time $t$):

$$
U_t=R_t+\gamma R_{t+1}+\gamma ^2 R_{t+2}+\gamma ^3 R_{t+3}+\cdots + \gamma ^{n-t} R_{n}
$$

- at the end of the game, we observe $u_t$, we observe all the rewards, $r_t,r_{t+1},\cdots,r_n$, thereby we know the discounted return

$$
u_t=r_t+\gamma r_{t+1}+\gamma^2r_{t+2}+\cdots+\gamma^{n-t}r_n
$$

在时刻 $t$，奖励 $R_t,\cdots,R_n$ 是**随机**的，所以回报 $U_t$ 是**随机**的。

- reward $R_i$ depends on $S_i$ and $A_i$.
- states can be random: $S_i$ ~ $p(\cdot|s_{i-1},a_{i-1})$.
- actions can be random: $A_i$ ~ $\pi(\cdot|s_i)$.
- if either $S_i$ or $A_i$ is random, then $R_i$ is random
- **so** $U_t$ depends on $S_t, A_t, S_{t+1}, A_{t+1}, \cdots, S_n, A_n$.

**Definition of Action-value function**
$$
Q_\pi(s_t,a_t)=\mathbb{E}[U_t|S_t=s_t,A_t=a_t]
$$
在这个动作价值函数的定义式中：

- $U_t$ depends on states $S_t, S_{t+1},\cdots,S_n$ and actions $A_t, A_{t+1},\cdots,A_n$. 
- regard $s_t$ and $a_t$ as observed values. 
- regard $S_{t+1},\cdots,S_n$ and $A_{t+1},\cdots,A_n$ as random variables. 
- $\mathbb{E}$ 积分求期望去掉了随机变量的随机性，求得了确定的期望值，即是动作价值函数值
- $Q_\pi(s_t,a_t)$ 当然也是跟策略函数 $\pi$ 有关的
- Action-value is the function of the state-action pair $(s,a)$, it **depends on** policy $\pi$.

**Definition of State-value function**
$$
V_\pi(s_t)=\mathbb{E}_A[Q_\pi(s_t,A)]
\\
A\  \sim \ \pi(\cdot|s_t)
$$
ok

- $V_\pi(s_t)=\mathbb{E}_A[Q_\pi(s_t,A)]=\sum_a \pi(a|s_t)\cdot Q_\pi(s_t,a)$.  --Actions are discrete. 
- $V_\pi(s_t)=\mathbb{E}_A[Q_\pi(s_t,A)]=\int \pi(a|s_t)\cdot Q_\pi(s_t,a) da$.  --Actions are continuous. 
- State value is the function of state $s$, it **depends on** policy $\pi$. 

**Understanding the Value Functions**

- **Action-value function**: $Q_\pi(s,a)=\mathbb{E}[U_t|S_t=s,A_t=a]$. 

​    **Given** policy $\pi$, $Q_\pi(s, a)$ evaluates **how good** it is for an agent to pick **action $a$** while being in state **$s$**. 

- **State-value functioin**: $V_\pi(s)=\mathbb{E}_A[Q_\pi(s,A)]$

​    For fixed policy $\pi$, $V_\pi(s)$ evaluates **how good** the **situation is** in state **$s$**.

​    $\mathbb{E}_s[V_\pi(S)]$ evaluates **how good** the **policy $\pi$** is. 