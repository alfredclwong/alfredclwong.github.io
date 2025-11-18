---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
---

[Email](mailto:alfred.cl.wong@gmail.com) \| [GitHub](https://github.com/alfredclwong) \| [LinkedIn](https://www.linkedin.com/in/alfred--wong)

<section>
  <h2>Blog</h2>
  <ul>
    {% assign blog_posts = site.categories.blog %}
    {% for post in blog_posts %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
</section>

<section>
  <h2>Theory</h2>
  <ul>
    {% assign theory_posts = site.categories.theory %}
    {% for post in theory_posts %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
</section>

<section>
  <h2>Paper Reviews</h2>
  <ul>
    {% assign review_posts = site.categories.review %}
    {% for post in review_posts %}
      <li><a href="{{ post.url }}">{{ post.title }}</a></li>
    {% endfor %}
  </ul>
</section>

