#!/usr/bin/env python3
"""Rebuild an inspected archive in place without importing bundled code.
Use --archive PATH. Backups go outside the repository; output is validated first.
"""
import argparse
import ast
import csv
import io
import re
import shutil
import tempfile
from pathlib import Path
from zipfile import ZipFile, ZipInfo, ZIP_DEFLATED

README = '''# SQL 练习台与只读取数工具

这是一个使用合成数据的本地 MCP 项目，不包含模型或 API 凭据。在线练习台与本地工具是两个使用入口。

## 运行

需要 Python 3.11 或更新版本。在解压后的项目根目录运行：

```bash
python -m pip install "mcp[cli]"
python quagent/server/smoke_mcp.py
```

自测通过说明工具层的列举、查询和示例拒绝测试通过，不代表所有查询正确。

在支持 MCP 的客户端里配置 stdio 服务：命令为 `python`，参数为你自己电脑上 `quagent/server/mcp_server.py` 的路径。不要将完成配置后的私人路径或凭据提交到公开仓库。

## 包含什么

- `quagent/server/`：查看场景、表结构、搜索字段、指标定义、检查 SQL 和执行查询的六项工具。
- `quagent/scenario/`：六张表的合成 SQLite 数据库、指标文档、24 道题及数据生成脚本。示例库不含 events 表。
- `workbench/sqlkit.py`：语句检查、只读连接、返回行数与超时控制。
- `quagent/eval/`：结果比对脚本和历史提交记录。历史记录只包含 Q01，不应解释为模型已经答对全部题目。

本地 run_sql 默认返回 100 行，最多 200 行，超时设置为 5 秒。使用前请核对指标口径；可执行的 SQL 也可能答错问题。

## 数据与验证

所有数据均为合成数据，不含真实业务记录。SQLite 不等同于 Hive 或 Spark，规则命中不等于确认作弊。

Q20 按 gmv 字段统计商品原价；Q21 按 user_id 去重统计命中用户，不能统计用户与小时组合数。当前数据上，命中用户为 159，命中小时内订单为 1643。

24 题在当前数据上的结果值已用独立 Python 聚合复算。该检查不覆盖所有边界数据、并列排序或模型作答能力。历史提交记录保留原样，不把参考查询冒充用户提交。

## 重新生成数据

在 `quagent/` 目录运行下面的命令会覆盖数据库，仅在保留备份后使用：

```bash
python scenario/generate_data.py --db scenario/anticheat.db --seed 20261105 --force
python scenario/build_questions.py --db scenario/anticheat.db
```

## 使用示例

先查看场景和指标定义，再检查并执行：

```sql
SELECT city, COUNT(*) AS orders,
       ROUND(SUM(pay_amt - refund_amt), 2) AS net_gmv
FROM orders
WHERE order_ts >= '2026-10-26' AND order_ts < '2026-11-02'
  AND status = 'delivered'
GROUP BY city
ORDER BY net_gmv DESC;
```

这里按订单表 city 分组；若题目指定用户所在城市，应关联 users 表。比对是比较结果表，不是比较 SQL 文本，也不是对业务结论的保证。
'''
INSTALL = '''# MCP 客户端接入

从解压根目录安装依赖并运行 START_HERE.md 中的自测命令。
在客户端自己的配置界面添加 stdio MCP 服务：命令 python，参数为解压后 quagent/server/mcp_server.py 的本机路径。
不同客户端配置格式可能不同，请依其官方文档配置。本包不附带个人电脑配置或凭据。
'''
PATH = re.compile(r'(?<![A-Za-z0-9])[A-Za-z]:[/\\]|/Users/|/home/')
PRIVATE = ['求职准备','严肃的工作区','弘(系统)','实实习习']
TEXT={'.md','.py','.json','.csv','.yml','.txt'}
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--archive',type=Path,required=True);args=ap.parse_args()
    source=args.archive
    backup=Path(tempfile.mkdtemp(prefix='quagent-backup-'))/'original.zip';shutil.copy2(source,backup)
    entries=[]
    with ZipFile(source) as z:
        for n in z.namelist():
            data=z.read(n)
            if n.endswith(('/README.md','/START_HERE.md','/DEMO.md')):data=README.encode()
            elif n.endswith('/harness/install.md'):data=INSTALL.encode()
            elif n.endswith('/harness/cordis.snippet.yml'):
                data=b'# Configure command python and the extracted server path in your own MCP client.\n'
            elif n.endswith('/eval/errors.md'):
                data=b'# Historical development notes excluded from the public package.\n'
            elif n.endswith('/server/instructions.md'):
                data='''使用合成数据回答当前场景问题。先查看场景、表结构和指标定义，确认时间范围与统计粒度。先调用 validate_sql，再调用 run_sql。保留只读约束；结果截断时应说明限制，不得把规则命中说成确认作弊。不要请求凭据或修改数据库。'''.encode()
            elif n.endswith('/scenario/build_questions.py'):
                s=data.decode();s=s.replace('ROUND(SUM(o.pay_amt), 2) AS gmv','ROUND(SUM(o.gmv), 2) AS gmv').replace('ROUND(SUM(o.pay_amt) / (SELECT SUM(pay_amt) FROM orders), 4) AS share','ROUND(SUM(o.gmv) / (SELECT SUM(gmv) FROM orders), 4) AS share').replace('SELECT COUNT(*) AS users, SUM(n) AS orders','SELECT COUNT(DISTINCT user_id) AS users, SUM(n) AS orders');data=s.encode()
            elif n.endswith('/scenario/questions.csv'):
                rows=list(csv.DictReader(io.StringIO(data.decode('utf-8-sig'))));fields=list(rows[0])
                for row in rows:
                    if row['id']=='Q20':row['golden_sql']=row['golden_sql'].replace('o.pay_amt','o.gmv').replace('SUM(pay_amt)','SUM(gmv)')
                    if row['id']=='Q21':row['golden_sql']=row['golden_sql'].replace('COUNT(*) AS users','COUNT(DISTINCT user_id) AS users')
                stream=io.StringIO();w=csv.DictWriter(stream,fieldnames=fields);w.writeheader();w.writerows(rows);data=stream.getvalue().encode()
            elif n.endswith('/eval/check_answers.py'):
                s=data.decode();s=s.replace('return out.get("columns", []), out.get("rows", []), ""','if out.get("truncated"):\n        return None, None, "Result truncated; full comparison unavailable"\n    return out.get("columns", []), out.get("rows", []), ""');s=s.replace('args.golden,','Path(args.golden).name,');data=s.encode()
            if Path(n).suffix in TEXT:
                s=data.decode()
                if PATH.search(s) or any(x in s for x in PRIVATE):raise SystemExit('Private marker in '+n)
                if n.endswith('.py'):ast.parse(s)
            entries.append((n,data))
    dest=source.with_suffix('.new.zip')
    if dest.exists():raise SystemExit('Output exists')
    with ZipFile(dest,'w',ZIP_DEFLATED) as z:
        for n,data in entries:
            i=ZipInfo(n,(2026,11,16,0,0,0));i.compress_type=ZIP_DEFLATED;i.external_attr=0o644<<16;z.writestr(i,data)
    with ZipFile(dest) as z:assert z.testzip() is None
    dest.replace(source);print('Rebuilt archive:',len(entries),'files; text privacy and syntax gates passed')
if __name__=='__main__':main()
