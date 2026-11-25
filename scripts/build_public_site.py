"""Build a public-only Pages artifact, excluding local/internal materials."""
from pathlib import Path
import shutil
import re
import json
from urllib.parse import unquote

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'_site_public'
if OUT.exists():
    raise SystemExit('Output already exists; inspect and remove it before rebuilding.')
OUT.mkdir()
for name in ['index.html','css','js','assets','blog','files','work']:
    src=ROOT/name
    if src.is_dir():shutil.copytree(src,OUT/name)
    else:shutil.copy2(src,OUT/name)
# Only indexed Blog pages are public; remove legacy exports and nested backups too.
index = (ROOT / 'blog' / 'posts.js').read_text(encoding='utf-8')
posts = json.loads(index.split('window.BLOG_POSTS = ', 1)[1].strip().removesuffix(';'))
allowed_blog = {'posts.js'} | {unquote(Path(post['url']).name) for post in posts}
for p in (OUT / 'blog').rglob('*'):
    if p.is_file() and (p.parent != OUT / 'blog' or p.name not in allowed_blog):
        p.unlink()
# No draft artifacts, Markdown sources, tests, local notes, or developer configuration.
for p in sorted(OUT.rglob('*'),key=lambda x:len(x.parts),reverse=True):
    if not p.is_file():continue
    if (p.suffix in {'.md','.py','.mjs'} or 'tests' in p.parts or
        'review' in p.name.lower() and p.parent==OUT/'files'/'work'):
        p.unlink()
# Notebook exports can contain local execution paths; redact only drive-root strings.
pattern=re.compile(r'(?<![A-Za-z0-9])[A-Za-z]:[/\\][^\r\n\"\'<>`]*')
count=0
for p in OUT.rglob('*'):
    if not p.is_file() or p.suffix not in {'.html','.js','.css','.json','.txt'}:continue
    text=p.read_text(encoding='utf-8')
    text,n=pattern.subn('[local-path-removed]',text)
    if n:p.write_text(text,encoding='utf-8');count+=n
# Content-version local JS/CSS references so a new HTML cannot reuse stale assets.
import hashlib
from urllib.parse import urlsplit
for page in OUT.rglob('*.html'):
    text=page.read_text(encoding='utf-8')
    def version_asset(match):
        prefix,url,suffix=match.groups()
        parsed=urlsplit(url)
        if parsed.scheme or parsed.netloc or not parsed.path.endswith(('.js','.css')):
            return match.group(0)
        target=(OUT/parsed.path.lstrip('/')) if parsed.path.startswith('/') else (page.parent/parsed.path)
        target=target.resolve()
        if not target.is_relative_to(OUT.resolve()) or not target.is_file():
            return match.group(0)
        digest=hashlib.sha256(target.read_bytes()).hexdigest()[:12]
        return prefix+parsed.path+'?v='+digest+suffix
    text=re.sub(r'((?:src|href)=\")([^\"]+)(\")',version_asset,text)
    page.write_text(text,encoding='utf-8')
print('Public-only artifact created; local path occurrences redacted:',count)
