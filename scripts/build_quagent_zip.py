#!/usr/bin/env python3
"""构建可下载的只读取数助手完整项目。

用法：
  python scripts/build_quagent_zip.py
  python scripts/build_quagent_zip.py --source "D:/.../quagent" --out files/work/quagent-local.zip
"""

import argparse
import gzip
import io
import json
import sys
import zipfile
from pathlib import Path

SITE = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path(
    "D:/弘(系统)/Documents/实实习习/业务能力1-大数据（SQL、Hive、Spark）"
    "/求职准备/quagent"
)
DEFAULT_OUT = SITE / "files" / "work" / "quagent-local.zip"
DEMO_GZ = SITE / "work" / "quagent" / "data" / "anticheat-demo.db.gz"

ROOT_DOCS = ["BUILD.md", "DEMO.md", "README.md", "PRIVACY_AND_REVIEW.md"]
SERVER_FILES = ["mcp_server.py", "smoke_mcp.py", "instructions.md"]
SCENARIO_FILES = [
    "generate_data.py",
    "build_questions.py",
    "questions.csv",
    "metrics.md",
    "labels.csv",
]
HARNESS_FILES = ["cordis.snippet.yml", "install.md"]
EVAL_FILES = ["check_answers.py", "errors.md", "scoreboard.md", "answers/Q01.sql"]

START_HERE = """# 只读取数助手 · 本地完整项目

这个包里的 `quagent/` 是真正的取数助手。模型通过 MCP 调用六个只读工具，可以回答任意业务问题；`questions.csv` 里的 24 道题只是评测集。

## 目录

- `quagent/server/` 是 MCP 服务器，暴露 `get_scenario`、`get_schema`、`search_columns`、`get_metrics`、`validate_sql`、`run_sql` 六个工具。
- `quagent/scenario/` 是场景说明、指标定义、题库，以及一份 6 张表的示例库 `anticheat.db`。
- `workbench/sqlkit.py` 是查询护栏，必须和 `quagent/` 保持同级，服务器启动时会按相对路径找到它。
- `quagent/harness/` 是挂载到 Harness 的配置片段和步骤。
- `quagent/eval/` 是评分脚本与当前进度。

## 运行

需要 Python 3.11 或更高版本。

1. 安装依赖。

   ```bash
   python -m pip install "mcp[cli]"
   ```

2. 自测工具层。

   ```bash
   python quagent/server/smoke_mcp.py
   ```

   看到 `ALL PASS` 说明只读工具、写操作拦截和多语句拦截都正常。

3. 挂载到 Harness。打开 `quagent/harness/install.md`，把 `cordis.snippet.yml` 里的路径改成你解压后的实际路径，追加到配置里，重启 Harness。

4. 在 Harness 里直接问业务问题，例如「上周各城市净 GMV 是多少」。模型会先读场景和指标，再写 SQL，先校验再执行。

## 关于示例库

示例库删掉了 `events` 表，把体积压到约 3.5MB，24 道题都不依赖它。需要完整的行为日志时，在 `quagent/` 目录运行：

```bash
python scenario/generate_data.py --db scenario/anticheat.db --seed 20261105 --force
```

随机种子固定，重建后数据与原项目一致。`--force` 会覆盖当前的示例库。

## 边界

数据全部为合成数据。规则命中不等于确认作弊。SQLite 方言不是 Hive/Spark，日期函数和分区写法迁移时要改写。
"""


def zip_info(name):
    info = zipfile.ZipInfo(name, date_time=(2026, 11, 15, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    return info


def add_bytes(zf, name, data):
    zf.writestr(zip_info(name), data, compresslevel=9)


def add_file(zf, base, rel, arcname=None):
    path = base / rel
    if not path.is_file():
        raise FileNotFoundError(path)
    add_bytes(zf, "quagent-local/" + (arcname or rel), path.read_bytes())


def filtered_schema(source):
    doc = json.loads((source / "scenario" / "schema_doc.json").read_text(encoding="utf-8"))
    doc["tables"].pop("events", None)
    doc["join_keys"] = [k for k in doc["join_keys"] if not k.startswith("events.")]
    return json.dumps(doc, ensure_ascii=False, indent=2).encode("utf-8")


def build(source, out):
    if not source.is_dir():
        raise SystemExit("源目录不存在：" + str(source))
    if not DEMO_GZ.is_file():
        raise SystemExit("示例库不存在：" + str(DEMO_GZ))

    out.parent.mkdir(parents=True, exist_ok=True)
    db_bytes = gzip.decompress(DEMO_GZ.read_bytes())

    with zipfile.ZipFile(out, "w") as zf:
        add_bytes(zf, "quagent-local/START_HERE.md", START_HERE.encode("utf-8"))
        for name in ROOT_DOCS:
            add_file(zf, source, name)
        for name in SERVER_FILES:
            add_file(zf, source, "server/" + name, "quagent/server/" + name)
        for name in SCENARIO_FILES:
            add_file(zf, source, "scenario/" + name, "quagent/scenario/" + name)
        add_bytes(zf, "quagent-local/quagent/scenario/schema_doc.json", filtered_schema(source))
        add_bytes(zf, "quagent-local/quagent/scenario/anticheat.db", db_bytes)
        for name in HARNESS_FILES:
            add_file(zf, source, "harness/" + name, "quagent/harness/" + name)
        for name in EVAL_FILES:
            add_file(zf, source, "eval/" + name, "quagent/eval/" + name)
        add_file(zf, source.parent, "workbench/sqlkit.py")

    return out


def main():
    parser = argparse.ArgumentParser(description="构建 quagent-local.zip")
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    out = build(args.source, args.out)
    size = out.stat().st_size
    with zipfile.ZipFile(out) as zf:
        names = zf.namelist()
    print("写出 " + str(out))
    print("文件数 %d，大小 %.1f MB" % (len(names), size / 1048576))
    for name in names:
        print("  " + name)
    return 0


if __name__ == "__main__":
    sys.exit(main())
