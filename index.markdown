---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
---

[Email](mailto:alfred.cl.wong@gmail.com) \| [GitHub](https://github.com/alfredclwong) \| [LinkedIn](https://www.linkedin.com/in/alfred--wong)

<section>
  <h2><a href="/blog/">Blog</a></h2>
  <ul>
    {% assign blog_posts = site.categories.blog | slice: 0, 5 %}
    {% for post in blog_posts %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
</section>

<section>
  <h2><a href="/theory/">Theory</a></h2>
  <ul>
    {% assign theory_posts = site.categories.theory | slice: 0, 5 %}
    {% for post in theory_posts %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
</section>
