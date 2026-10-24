#!/usr/bin/env python3
"""Build blog pages and the post index from Markdown files in blog/.

Each post is a Markdown file with YAML front matter:

    ---
    title: My Post
    date: 2026-10-23
    summary: One-line description
    tags: [Essay, Notes]
    ---

Body text follows. The file name (or the `slug` field) becomes the URL.

Usage:
    python scripts/build_blog.py
"""

import datetime as dt
import html
import json
import re
import sys
from pathlib import Path
from urllib.parse import quote

import markdown
import yaml

ROOT = Path(__file__).resolve().parents[1]
BLOG = ROOT / "blog"
TEMPLATE = ROOT / "scripts" / "post_template.html"
CONFIG = ROOT / "js" / "config.js"
MARKER = 'content="scripts/build_blog.py"'


def site_name():
    text = CONFIG.read_text(encoding="utf-8")
    match = re.search(r'name\s*:\s*"([^"]+)"', text)
    return match.group(1) if match else "Personal Site"


def read_frontmatter(path):
    text = path.read_text(encoding="utf-8")
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            meta = yaml.safe_load(parts[1]) or {}
            if isinstance(meta, dict):
                return meta, parts[2].lstrip("\n")
    return {}, text


def slugify(value):
    return re.sub(r"[^a-zA-Z0-9]+", "-", str(value)).strip("-").lower()


def normalize_date(value):
    if isinstance(value, dt.datetime):
        return value.date().isoformat()
    if isinstance(value, dt.date):
        return value.isoformat()
    text = str(value or "").strip()
    return text if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text) else ""


def protect(text):
    """Swap code spans and math for tokens so Markdown does not rewrite them."""
    store = {}
    counter = [0]

    def token(kind, raw):
        key = "@@%s%d@@" % (kind, counter[0])
        counter[0] += 1
        store[key] = (kind, raw)
        return key

    chunks = re.split(r"(?ms)(^```.*?^```)", text)
    out = []
    for i, chunk in enumerate(chunks):
        if i % 2 == 1:
            out.append(chunk)
            continue
        chunk = re.sub(r"`[^`\n]+`", lambda m: token("CODE", m.group(0)), chunk)
        chunk = re.sub(r"\$\$(.+?)\$\$", lambda m: token("DMATH", m.group(1)), chunk, flags=re.S)
        chunk = re.sub(r"(?<!\\)\$([^$\n]+?)(?<!\\)\$", lambda m: token("IMATH", m.group(1)), chunk)
        out.append(chunk)
    return "".join(out), store


def restore(html_text, store):
    for key, (kind, raw) in store.items():
        if kind == "DMATH":
            value = "$$" + html.escape(raw, quote=False) + "$$"
        elif kind == "IMATH":
            value = "$" + html.escape(raw, quote=False) + "$"
        else:
            value = html.escape(raw, quote=False)
        html_text = html_text.replace(key, value)
    return html_text


def render_markdown(body):
    protected, store = protect(body)
    md = markdown.Markdown(
        extensions=["fenced_code", "codehilite", "tables", "footnotes", "sane_lists", "attr_list"],
        extension_configs={"codehilite": {"guess_lang": False, "linenums": False}},
    )
    return restore(md.convert(protected), store)


def build():
    template = TEMPLATE.read_text(encoding="utf-8")
    name = site_name()
    year = dt.date.today().year

    posts = []
    used = set()
    for path in sorted(BLOG.glob("*.md")):
        meta, body = read_frontmatter(path)
        if meta.get("draft"):
            continue

        title = str(meta.get("title") or path.stem)
        date = normalize_date(meta.get("date") or meta.get("publish_date"))
        if not date:
            print("warning: %s has no valid date, using 1970-01-01" % path.name, file=sys.stderr)
            date = "1970-01-01"

        summary = str(meta.get("summary") or meta.get("desc") or meta.get("description") or "")
        tags = meta.get("tags") or meta.get("topics") or []
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",")]
        tags = [str(t).strip() for t in tags if str(t).strip()]

        base = slugify(meta.get("slug") or path.stem) or "post-" + date
        slug = base
        n = 2
        while slug in used:
            slug = "%s-%d" % (base, n)
            n += 1
        used.add(slug)

        content = render_markdown(body)
        tag_html = "".join('<span class="post-tag">%s</span>' % html.escape(t) for t in tags)
        page = (template
                .replace("{{TITLE}}", html.escape(title))
                .replace("{{DESC}}", html.escape(summary))
                .replace("{{DATE}}", html.escape(date))
                .replace("{{TAGS}}", tag_html)
                .replace("{{CONTENT}}", content)
                .replace("{{SITE_NAME}}", html.escape(name))
                .replace("{{YEAR}}", str(year)))
        out = BLOG / ("%s.html" % slug)
        out.write_text(page, encoding="utf-8")

        posts.append({
            "title": title,
            "date": date,
            "desc": summary,
            "tags": tags,
            "url": "blog/%s.html" % quote(slug),
        })
        print("built: %s" % out.relative_to(ROOT))

    posts.sort(key=lambda p: (p["date"], p["title"]), reverse=True)

    expected = {Path(p["url"]).name for p in posts}
    for old in BLOG.glob("*.html"):
        if old.name not in expected and MARKER in old.read_text(encoding="utf-8", errors="ignore"):
            old.unlink()
            print("removed stale: %s" % old.relative_to(ROOT))

    data = json.dumps(posts, ensure_ascii=False, indent=2).replace("</", "<\\/")
    (BLOG / "posts.js").write_text(
        "// Generated by scripts/build_blog.py — do not edit.\nwindow.BLOG_POSTS = %s;\n" % data,
        encoding="utf-8",
    )
    print("built: blog/posts.js (%d posts)" % len(posts))


if __name__ == "__main__":
    build()
