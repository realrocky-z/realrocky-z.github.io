"""Build a public-only Pages artifact, excluding local/internal materials."""
from pathlib import Path
import shutil
import re

ROOT=Path(__file__).resolve().parent.parent
OUT=ROOT/'_site_public'
if OUT.exists():
    raise SystemExit('Output already exists; inspect and remove it before rebuilding.')
OUT.mkdir()
for name in ['index.html','css','js','assets','blog','files','work']:
    src=ROOT/name
    if src.is_dir():shutil.copytree(src,OUT/name)
    else:shutil.copy2(src,OUT/name)
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
print('Public-only artifact created; local path occurrences redacted:',count)
