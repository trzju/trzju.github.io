---
layout: default
title: Home
---

## 欢迎

这里是 Trzju 的个人技术博客。主要分享关于 Robot Operating System (ROS)、强化学习 (RL) 的算法笔记以及一些摄影作品。

### 最新文章

<ul>
  {% for post in site.posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      <span style="color: #999; font-size: 0.8em;"> - {{ post.date | date: "%Y-%m-%d" }}</span>
    </li>
  {% endfor %}
</ul>
