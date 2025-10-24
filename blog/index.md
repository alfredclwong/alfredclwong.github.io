---
layout: default
title: Blog
---

<h1>{{ page.title }}</h1>

<ul>
  {% assign blog_posts = site.posts | where: "category", "blog" %}
  {% for post in blog_posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      <span class="date">{{ post.date | date: "%B %-d, %Y" }}</span>
    </li>
  {% endfor %}
</ul>