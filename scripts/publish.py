#!/usr/bin/env python3
"""把 Markdown 文章或 PDF 作品发布到个人主页。

用法:
  python scripts/publish.py blog "D:/obsidian/40 发布/已发布/博客/文章.md"
  python scripts/publish.py pdf "D:/obsidian/40 发布/已发布/讲义.pdf" \
      --title "标题" --desc "简介" --section notes --tags 统计,课程

Blog 文章会复制到 blog/ 并立即重新构建文章页与列表。
PDF 会做 AES-256 文件级加密，密码从环境变量 SITE_PDF_PASSWORD 读取，
或用 --password 传入；密码不会写进网站配置。
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
CONFIG = SITE / "js" / "config.js"
BLOG_DIR = SITE / "blog"
FILES_DIR = SITE / "files"
ASSETS_DIR = SITE / "assets"


def find_array(src: str, key: str):
    marker = f"{key}: ["
    start = src.index(marker)
    i = start + len(marker)
    depth, quote, esc = 1, None, False
    while i < len(src):
        c = src[i]
        if quote:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == quote:
                quote = None
        else:
            if c in "\"'":
                quote = c
            elif c == "[":
                depth += 1
            elif c == "]":
                depth -= 1
                if depth == 0:
                    return start + len(marker), i
        i += 1
    raise ValueError(f"找不到数组 {key}")


def insert_entry(config_text: str, key: str, entry: str) -> str:
    begin, _ = find_array(config_text, key)
    return config_text[:begin] + "\n" + entry + config_text[begin:]


def js_entry(fields: dict) -> str:
    lines = ["        {"]
    items = list(fields.items())
    for idx, (k, v) in enumerate(items):
        comma = "," if idx < len(items) - 1 else ""
        if isinstance(v, list):
            value = "[" + ", ".join(json.dumps(x, ensure_ascii=False) for x in v) + "]"
        else:
            value = json.dumps(v, ensure_ascii=False)
        lines.append(f"            {k}: {value}{comma}")
    lines.append("        },")
    return "\n".join(lines)


def site_name() -> str:
    m = re.search(r'name:\s*"([^"]+)"', CONFIG.read_text(encoding="utf-8"))
    return m.group(1) if m else "Personal Site"


def publish_blog(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from build_blog import build, read_frontmatter, slugify

    src = Path(args.markdown).expanduser().resolve()
    if not src.is_file():
        sys.exit(f"找不到文件：{src}")

    meta, _ = read_frontmatter(src)
    title = str(meta.get("title") or src.stem)
    date = str(meta.get("date") or meta.get("publish_date") or dt.date.today().isoformat())
    base = slugify(args.slug or meta.get("slug") or src.stem) or f"post-{date}"
    dest = BLOG_DIR / f"{base}.md"

    text = src.read_text(encoding="utf-8")
    if dest.resolve() == src:
        print(f"文章已在 blog/{dest.name}，直接重新构建")
    elif dest.exists():
        if dest.read_text(encoding="utf-8") != text:
            sys.exit(f"blog/{dest.name} 已存在且内容不同；直接编辑该文件，或用 --slug 换个名字")
        print(f"blog/{dest.name} 已是最新")
    elif args.dry_run:
        print(f"[dry-run] 会复制到 blog/{dest.name}")
    else:
        BLOG_DIR.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        print(f"已复制到 blog/{dest.name}")

    if args.dry_run:
        print(f"[dry-run] 会生成 blog/{base}.html 并刷新 blog/posts.js")
        return

    build()
    print(f"已生成 blog/{base}.html，并刷新 Blog 列表：{title}")


def publish_pdf(args):
    src = Path(args.pdf).expanduser().resolve()
    password = args.password or os.environ.get("SITE_PDF_PASSWORD")
    if not password:
        sys.exit("缺少 PDF 密码：设置环境变量 SITE_PDF_PASSWORD 或传 --password")
    base = re.sub(r"[^a-zA-Z0-9]+", "-", (args.slug or src.stem)).strip("-").lower() or "document"
    dest = FILES_DIR / f"{base}.pdf"
    image = args.image or "assets/doc-placeholder.svg"
    tags = [t.strip() for t in args.tags.split(",") if t.strip()] if args.tags else []
    entry = js_entry({
        "title": args.title, "desc": args.desc, "tags": tags,
        "file": f"files/{base}.pdf", "image": image,
    })
    if args.dry_run:
        print(f"[dry-run] 会加密复制到 files/{base}.pdf，并在 config.js 的 {args.section} 中插入：\n{entry}")
        return
    import fitz
    doc = fitz.open(src)
    doc.save(dest, encryption=fitz.PDF_ENCRYPT_AES_256, owner_pw=password,
             user_pw=password, permissions=fitz.PDF_PERM_PRINT)
    doc.close()
    text = CONFIG.read_text(encoding="utf-8")
    CONFIG.write_text(insert_entry(text, args.section, entry), encoding="utf-8")
    print(f"已加密复制到 files/{base}.pdf")
    print(f"已更新 js/config.js 的 {args.section} 数组")


def main():
    ap = argparse.ArgumentParser(description="发布作品到个人主页")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("blog", help="把 Markdown 复制进 blog/ 并重新构建")
    b.add_argument("markdown")
    b.add_argument("--slug", default="")
    b.add_argument("--dry-run", action="store_true")
    b.set_defaults(func=publish_blog)

    p = sub.add_parser("pdf", help="加密 PDF 并加入 Learning 卡片")
    p.add_argument("pdf")
    p.add_argument("--title", required=True)
    p.add_argument("--desc", default="")
    p.add_argument("--tags", default="")
    p.add_argument("--section", choices=["notes", "tutoring"], default="notes")
    p.add_argument("--slug", default="")
    p.add_argument("--image", default="")
    p.add_argument("--password", default="")
    p.add_argument("--dry-run", action="store_true")
    p.set_defaults(func=publish_pdf)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
