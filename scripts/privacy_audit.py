"""Report local-path locations without printing matched values; skip auth/config files."""
import re
import sys
import zipfile
from pathlib import Path
PATH = re.compile(r'(?:[A-Za-z]:[/\\]|/Users/|/home/|file:///)', re.I)
TEXT = {'.md', '.html', '.js', '.py', '.yml', '.yaml', '.json', '.txt', '.csv', '.toml', '.css'}
SKIP = {'.git', '.claude', 'node_modules', '__pycache__'}
def scan(name, data):
    lines = data.decode('utf-8', 'replace').splitlines()
    hits = [str(i + 1) for i, line in enumerate(lines) if PATH.search(line)]
    if hits:
        print(name, 'absolute-path lines:', ','.join(hits))
def main():
    for p in sorted(Path(sys.argv[1]).rglob('*')):
        if not p.is_file() or any(x in SKIP or x.startswith('.qa-') for x in p.parts):
            continue
        if p.suffix == '.zip':
            with zipfile.ZipFile(p) as z:
                for n in z.namelist():
                    if Path(n).suffix in TEXT:
                        scan(p.as_posix() + '::' + n, z.read(n))
        elif p.suffix in TEXT:
            scan(p.as_posix(), p.read_bytes())
if __name__ == '__main__':
    main()
