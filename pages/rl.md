---
layout: archive
title: "Reinforcement Learning"
permalink: /rl/
author_profile: true
---

这里记录我的强化学习算法推导与实验笔记。

<ul>
  {% for post in site.categories.RL %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      <span style="font-size:0.8em; color:#888;"> ({{ post.date | date: "%Y-%m-%d" }})</span>
    </li>
  {% endfor %}
</ul>
