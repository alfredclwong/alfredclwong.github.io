---
layout: default
title: Paper Reviews
---

<h1>{{ page.title }}</h1>

<ul>
  {% assign review_posts = site.posts | where: "category", "review" %}
  {% for post in review_posts %}
    <li>
      <a href="{{ post.url }}">{{ post.title }}</a>
      <span class="date">{{ post.date | date: "%B %-d, %Y" }}</span>
    </li>
  {% endfor %}
</ul>