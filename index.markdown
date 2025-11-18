---
# Feel free to add content and custom Front Matter to this file.
# To modify the layout, see https://jekyllrb.com/docs/themes/#overriding-theme-defaults

layout: default
---

[Email](mailto:alfred.cl.wong@gmail.com) \| [GitHub](https://github.com/alfredclwong) \| [LinkedIn](https://www.linkedin.com/in/alfred--wong)

<section>
  <h2>Recent Posts</h2>
  <ul>
    {% assign recent_posts = site.posts | sort: 'date' | reverse %}
    {% for post in recent_posts limit:10 %}
      <li>
        <span class="date">{{ post.date | date: "%d %b %y" }}</span>
        <a href="{{ post.url }}">{{ post.title }}</a>
        <span class="category">[{{ post.category }}]</span>
        {% if post.subtitle %}<span class="subtitle">"{{ post.subtitle }}"</span>{% endif %}
        <span class="wordcount">({{ post.content | number_of_words }})</span>
      </li>
    {% endfor %}
  </ul>
</section>

