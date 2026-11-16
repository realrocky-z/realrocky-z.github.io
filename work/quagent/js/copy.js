/* ============================================================
   文案总表：页面里所有给人读的文字都在这里。
   改完保存，刷新 http://127.0.0.1:8766/work/quagent/ 就能看到。

   - 键名以 Html 结尾的值可以写 <code>、<strong> 等简单标签。
   - 文案里的 {name} 是占位符，由程序填入数字或名称。
   - 表结构、场景与坑点仍在 data/schema.json；标准答案在 data/questions.json。
   - 24 道题的题干也在本文件 questions 里，按 Q01…Q24 排列。
   ============================================================ */
(function (root) {
    root.QUAGENT_COPY = {

        meta: {
            title: "智能取数助手（SQL）｜在线练习",
            back: "← The Workshop",
            navTitle: "智能取数助手（SQL）",
            navSuffix: " · 在线练习",
            pdf: "一页纸说明（PDF）"
        },

        hero: {
            kicker: "合成数据 · 在线练习与本地工具",
            title: "智能取数助手（SQL）",
            leadHtml: "我把一组业务场景题做成了浏览器里的 SQL 练习台，同时为本地项目提供查看表结构、指标口径和执行只读查询的工具。两者使用合成数据，重点是练习从问题到查询的过程。",
            steps: [
                "<strong>查看口径。</strong>确认题目要求的时间范围、订单状态与指标定义。",
                "<strong>确认字段。</strong>查看表结构和关联关系，注意一行数据代表什么。",
                "<strong>编写查询。</strong>在编辑器中写 SQL；「检查」按钮只编译查询并返回执行计划，不取数。",
                "<strong>执行与比对。</strong>运行查询，查看结果及其与参考答案的差异。数值比较允许 0.01 的绝对误差。"
            ],
            baseline: "24 道题在当前数据上的结果值已用独立 Python 聚合复算，并修正 Q20 的 GMV 字段和 Q21 的用户去重。验证不覆盖所有边界数据或模型作答能力；下载包的历史提交记录仍仅包含 Q01。",
            position: "在线页面不调用模型，SQL 在浏览器内执行。本地项目另提供六个 MCP 工具，接入支持 MCP 的客户端后可由模型调用。在线练习和本地取数是两个使用入口。",
            howto: {
                summary: "开始练习",
                body: "加载示例数据后选择一道题，在编辑器中编写 SQL，点击「执行」查看结果。侧栏提供表结构、关联关系与指标定义；需要对照时可以查看参考写法。"
            }
        },

        tabs: {
            aria: "页面分区",
            workbench: "做题",
            scenario: "数据和指标",
            safety: "查询限制与验证"
        },

        workbench: {
            questionsTitle: "练习题",
            questionsSub: "24 道题来自同一个模拟场景。写 SQL，执行后按结果集和标准答案比对。",
            selectAria: "选择题目",
            reset: "清空做题记录",
            loadingTitle: "正在加载题目…",
            editorAria: "SQL 编辑器",
            placeholder: "在这里写一条查询，比如 SELECT city, COUNT(*) AS orders FROM orders GROUP BY city;",
            buttons: {
                validate: "检查",
                run: "执行",
                clear: "清空",
                solution: "参考写法"
            },
            dbStatus: "示例数据未加载",
            resultEmpty: "点「执行」之后，结果会显示在这里。",
            clearResultEmpty: "执行后会显示结果。"
        },

        sidebar: {
            load: {
                title: "示例数据",
                note: "六张表，压缩后大约 3.5MB。第一次点加载会下载这份数据，之后查询都在浏览器里完成，不会发到任何服务器。",
                button: "加载示例数据"
            },
            download: {
                title: "完整项目",
                lead: "在线版用于编写 SQL 和比对结果。本地项目提供六个 MCP 工具，接入支持 MCP 的客户端后可由模型调用。",
                button: "下载完整项目（约 3.6MB）",
                stepsSummary: "下载后怎么运行",
                steps: [
                    "解压，确认目录里有 <code>quagent/</code> 和 <code>workbench/</code> 两个文件夹。",
                    "安装依赖，运行 <code>python -m pip install \"mcp[cli]\"</code>。",
                    "运行 <code>python quagent/server/smoke_mcp.py</code>，看到 <code>ALL PASS</code> 说明工具层正常。",
                    "参考 <code>quagent/harness/install.md</code> 配置支持 MCP 的客户端，使用包内场景和工具进行取数。"
                ],
                fallback: "浏览器不支持进度时，点这里直接下载"
            },
            tables: {
                title: "六张表分别是什么",
                all: "查看每张表的全部字段"
            },
            joins: {
                title: "表之间怎么关联"
            },
            metrics: {
                title: "指标怎么算",
                all: "查看全部指标算法"
            }
        },

        scenario: {
            schema: {
                title: "先看数据长什么样",
                leadHtml: "写聚合之前先确认一行代表什么。订单表里一行是一单，设备表里一行是某个账号用过某台设备。同一台设备可以对应多个账号，所以算用户规模时要按 <code>user_id</code> 去重，不能数设备，也不能数订单行。"
            },
            metrics: {
                title: "每个指标怎么算"
            },
            joins: {
                title: "表之间怎么关联，哪里容易算错"
            },
            tools: {
                title: "本地项目提供的六个工具",
                lead: "下载包通过模型上下文协议（Model Context Protocol，MCP）提供以下接口。模型可调用工具查看场景、表结构与指标定义，再检查或运行查询；在线页面不连接模型。"
            },
            generated: {
                joinsHeading: "表之间怎么关联",
                pitfallsHeading: "容易算错的地方",
                demoNote: "为了减小下载体积，示例数据里没有 events 表。24 道练习题都用不到它，其余六张表和原项目一致。"
            },
            metricsHtml: `<h2>指标口径</h2>
<p>数据为合成数据；时间统一 UTC，金额统一为美元等值，参考日为 2026-11-01。</p>
<h3>规模</h3>
<ul>
<li><strong>GMV</strong>：<code>SUM(orders.gmv)</code>，商品原价口径。</li>
<li><strong>净 GMV</strong>：<code>SUM(orders.pay_amt - orders.refund_amt)</code>，扣退款后的实收口径。</li>
<li><strong>订单数</strong>：按 <code>order_id</code> 去重计数；除非问题明确说“全部订单”，默认只算 <code>status = 'delivered'</code>。</li>
<li><strong>客单价（AOV）</strong>：净 GMV ÷ 已送达订单数，保留两位小数。</li>
<li><strong>活跃用户</strong>：时间窗内至少有一笔已送达订单的去重 <code>user_id</code>。</li>
</ul>
<h3>新客与复购</h3>
<ul>
<li><strong>新客订单</strong>：<code>orders.is_new_user = 1</code>，即下单时注册未满 7 天。</li>
<li><strong>新客订单占比</strong>：新客订单数 ÷ 已送达订单数。</li>
<li><strong>7 日首单率</strong>：注册后 7 天内至少下过一单的账号数 ÷ 同期注册账号数；观察窗口不足 7 天的新注册账号不计入分母。</li>
<li><strong>复购</strong>：时间窗内已送达订单 ≥2 的单账号；复购率 = 复购账号数 ÷ 下单账号数。</li>
</ul>
<h3>优惠券</h3>
<ul>
<li><strong>领券数</strong>：<code>COUNT(coupons.coupon_id)</code>。</li>
<li><strong>核销数</strong>：<code>used_ts IS NOT NULL</code> 的券数。</li>
<li><strong>核销率</strong>：核销数 ÷ 领券数，保留四位小数。</li>
<li><strong>券滥用嫌疑</strong>：领券到核销间隔小于 60 秒，且核销订单状态为 <code>refunded</code>。这是规则命中，不是确认作弊。</li>
<li><strong>折扣强度</strong>：已核销订单的 <code>SUM(discount_amt) ÷ SUM(gmv)</code>。</li>
</ul>
<h3>退款</h3>
<ul>
<li><strong>退款率</strong>：<code>status = 'refunded'</code> 的订单数 ÷ 全部订单数；按商家或骑手分组时先设最小单量门槛，避免小样本噪声。</li>
<li><strong>退款金额</strong>：<code>SUM(refund_amt)</code>，只对 <code>refunded</code> 订单非零。</li>
</ul>
<h3>风险与作弊嫌疑</h3>
<p>规则命中只代表“需要人工复核”，不等于确认作弊。场景内可用的三类规则：</p>
<p>1. <strong>多账号设备</strong>：同一 <code>device_id</code> 关联 ≥3 个账号。</p>
<p>2. <strong>券滥用</strong>：60 秒内核销且订单退款。</p>
<p>3. <strong>频次异常</strong>：同一账号在同一小时内下单 ≥8 单。</p>
<p>自交易嫌疑：骑手的常用设备同时被某个账号使用，且该骑手配送了该账号的订单。</p>
<h3>常见口径错误</h3>
<ul>
<li>用 <code>gmv</code> 代替净 GMV，忽略退款。</li>
<li>数订单行而不是去重账号，虚增用户规模。</li>
<li>忘记排除 <code>cancelled</code> 和 <code>undelivered</code>。</li>
<li>把 <code>used_ts IS NULL</code> 写成 <code>used_ts = ''</code>。</li>
<li>用设备行数代替账号数。</li>
<li>小样本商家直接算退款率，没有最小单量门槛。</li>
</ul>`
        },

        safety: {
            title: "查询限制与验证范围",
            lead: "语句检查、只读连接和执行限制用于减少误操作，不能保证指标口径正确。在线版与本地版采用不同的执行方式，下面分别说明。",
            layers: [
                {
                    no: "01",
                    title: "合成数据",
                    bodyHtml: "练习数据为模拟生成，不包含真实公司、用户或交易信息。计算结果用于演示查询方法，不代表真实业务表现。",
                    evidenceHtml: "在线数据说明见「数据和指标」；本地数据说明见下载包的场景文档。"
                },
                {
                    no: "02",
                    title: "只读执行",
                    bodyHtml: "在线页面在执行前检查语句，查询超过 5 秒时终止 Worker，之后需要重新加载数据。本地连接使用 <code>PRAGMA query_only=ON</code>，并设置语句类型、返回行数和超时限制。",
                    evidenceHtml: "本地实现见下载包内 <code>workbench/sqlkit.py</code> 和 <code>quagent/server/mcp_server.py</code>。"
                },
                {
                    no: "03",
                    title: "结果比较",
                    bodyHtml: "在线页面比较结果列数、行数与单元格数值；是否比较行顺序由题目配置决定。结果区最多展示 200 行，最多取回 1000 行用于比对。参考结果一致不等于查询在其他数据上也正确。",
                    evidenceHtml: "比较规则见页面查询后的提示；本地评测脚本位于下载包的 <code>quagent/eval/check_answers.py</code>。"
                },
                {
                    no: "04",
                    title: "当前验证范围",
                    bodyHtml: "24 题在当前数据上的结果值已独立复算；排序、并列名次与边界反例尚未全面验证。下载包的历史提交仅包含 Q01。项目使用 SQLite，迁移到 Hive 或 Spark 需要调整语法并重新测试。",
                    evidenceHtml: "提交记录见下载包内 <code>quagent/eval/scoreboard.md</code>。"
                }
            ],
            examples: {
                title: "查看语句检查示例",
                lead: "点击示例会将语句填入编辑器并执行。前四条用于观察执行前的拒绝提示，不需要加载数据；后两条用于观察结果行数与超时限制。"
            },
            audit: {
                title: "操作记录",
                leadHtml: "在线页面在浏览器本地保存最近 30 次操作。本地工具的调用日志写入项目目录下的 <code>scenario/query-log.jsonl</code>。",
                empty: "还没有操作。"
            }
        },

        banner: {
            html: "<strong>这个页面需要从网站打开。</strong> 直接双击本地文件时，浏览器会禁止页面读取数据文件和 Web Worker。线上地址是 <code>https://realrocky-z.github.io/work/quagent/</code>。要在本机预览，请在 <code>个人主页/</code> 目录运行 <code>python -m http.server 8766</code>，再访问 <code>http://localhost:8766/work/quagent/</code>。"
        },

        modal: {
            choice: {
                title: "选择一种写法",
                body: "可查看预置的参考写法；若要连接模型，请下载项目并在本地配置客户端。",
                builtin: "看内置参考示例",
                apiKey: "了解本地接入"
            },
            notice: {
                title: "建议下载到本地使用",
                body: "本页面不接收 API Key。若要让模型调用取数工具，请下载项目并在本地配置支持 MCP 的客户端。",
                continue: "继续看示例",
                download: "下载完整项目",
                ok: "知道了"
            },
            overwrite: {
                title: "编辑器里已经有内容",
                body: "参考写法是预置的标准答案，不是模型现场生成，会覆盖编辑器内容。继续吗？",
                fill: "覆盖并填入",
                cancel: "取消"
            },
            reset: {
                title: "清空做题记录",
                body: "本地保存的做题进度会被清空，编辑器里的 SQL 不受影响。",
                clear: "清空",
                cancel: "取消"
            }
        },

        reference: {
            stop: "停止",
            loadFailTitle: "参考写法没有加载出来",
            loadFail: "原因是 {reason}。",
            noSolutionTitle: "这道题没有参考写法",
            noSolutionDetail: "换一道题再试。"
        },

        messages: {
            fileStatus: "file:// 无法加载数据",
            fileTitle: "请用本地 HTTP 服务打开",
            fileText: "浏览器不允许直接双击打开时读取页面需要的运行文件和数据。按页面底部的提示启动本地服务，就能正常使用。",
            dataFailTitle: "数据加载失败",
            dataFail: "{reason}。请确认通过本地 HTTP 服务打开页面。",
            questionTitle: "{category} · {id}",
            questionOption: "{id} · {text}",
            validateNoDbTitle: "先加载示例数据",
            validateNoDbDetail: "先点「加载示例数据」。你也可以先试一条一定会被拦住的语句。",
            validatePassTitle: "检查通过",
            validatePassDetail: "编译检查通过。下面是执行计划，尚未执行查询取数。",
            validateFailTitle: "检查没通过",
            blockedTitle: "已拦截",
            blockedDetail: "这条语句没有进入数据库。",
            runNoDbDetail: "示例数据压缩后大约 3.5MB，加载以后所有查询都在你自己的浏览器里完成。",
            passTitle: "通过 · {id}",
            timeoutTitle: "查询超时",
            timeoutDetail: "这条语句超过 5 秒还没有算完，页面把它停掉了，所以不会卡住。重新加载示例数据后可以继续。",
            timeoutAudit: "超过 5 秒，已中止",
            timeoutStatus: "查询已中止，请重新加载数据",
            runFailTitle: "执行失败",
            dbPreparing: "正在准备",
            dbReload: "重新加载示例数据",
            dbLoadFail: "加载失败，原因是 {reason}",
            dbRetry: "重试",
            dbDownloading: "正在下载 {pct}%",
            dbUnzip: "正在解压",
            dbGzip: "浏览器不支持 gzip 解压，请更新浏览器",
            dbReady: "示例数据已经准备好",
            dbTerminated: "已终止，请重新加载",
            workerUnknown: "未知错误",
            workerError: "Worker 错误",
            workerTimeout: "超过 {seconds} 秒未完成",
            workerKilled: "Worker 已终止",
            resultRowsLabel: "共",
            resultRowsValue: "{n} 行",
            resultRowsTruncatedValue: "{n} 行，页面只显示前 200 行",
            resultElapsedLabel: "用了",
            resultElapsedValue: "{ms} 毫秒",
            resultColumnsLabel: "返回",
            resultColumnsValue: "{n} 列",
            resultEmpty: "查询成功，但没有返回行。",
            resultLimitNote: "页面最多显示 200 行，判定时用的是前 1000 行。",
            planLabel: "执行计划（还没有真正取数）",
            planEmpty: "编译检查通过，但数据库没有返回执行计划。",
            planSql: "这条语句是 {sql}。",
            scenarioSentence: "{business}数据的时间范围是 {start} 到 {end}，时区是 {tz}，金额按{currency}计算。",
            sideRowCount: "{n} 行",
            sideTableDesc: "{grain}（{count} 行）",
            columnDesc: "{desc}（{type}）",
            loadNote: "六张表，其中 60,430 个订单、20,150 张优惠券，压缩后大约 {mb}MB。第一次点加载会下载这份数据，之后查询都在浏览器里完成，不会发到任何服务器。",
            joinTemplate: "{a}表的 {colA} 对应{b}表的 {colB}",
            downloadPreparing: "正在准备下载。",
            downloadProgress: "正在下载 {pct}%",
            downloadProgressMb: "正在下载 {mb}MB",
            downloadDone: "下载完成，解压后按下面的步骤运行。",
            downloadFail: "下载没有成功，原因是 {reason}。可以用下面的直接下载链接。",
            downloadAgain: "重新下载",
            downloadRetry: "重试下载"
        },

        labels: {
            difficulty: {
                easy: "简单",
                medium: "中等",
                hard: "较难"
            },
            tables: {
                users: "账号",
                devices: "设备",
                riders: "骑手",
                merchants: "商家",
                orders: "订单",
                coupons: "优惠券"
            },
            audit: {
                pass: "放行",
                deny: "拒绝"
            }
        },

        tools: [
            { name: "get_scenario", desc: "返回业务背景、时间范围、六张表的清单、表之间怎么关联，以及哪里容易算错。" },
            { name: "get_schema", desc: "返回每张表有哪些字段、字段是什么类型、是什么意思，各有多少行。也可以只查一张表。" },
            { name: "search_columns", desc: "不知道某个字段在哪张表里时，可以用关键词搜字段名和说明。" },
            { name: "get_metrics", desc: "返回成交额、退款后的成交额、客单价、新客、复购、优惠券核销、退款和作弊嫌疑这些指标各自怎么算。" },
            { name: "validate_sql", desc: "只检查语句能不能执行，不真的取数。写操作和一次提交多条语句都会被拒绝。" },
            { name: "run_sql", desc: "真正执行查询。默认返回 100 行，最多 200 行，超过 5 秒就中止。" }
        ],

        examples: [
            { title: "写操作", sql: "DROP TABLE orders", note: "这条语句在执行之前就被拦下，不需要加载数据。" },
            { title: "多语句", sql: "SELECT 1; DROP TABLE orders", note: "这里其实写了两条语句，而页面只接受一条。" },
            { title: "WITH + DELETE", sql: "WITH x AS (SELECT 1) DELETE FROM orders", note: "前面加上 WITH，也绕不过对写操作的检查。" },
            { title: "PRAGMA", sql: "PRAGMA table_info(orders)", note: "它既不是查询，也不是查看执行计划，所以不在允许范围内。" },
            { title: "结果太多", sql: "SELECT * FROM orders", note: "页面最多显示 200 行，判定时用前 1000 行。" },
            { title: "算不完", sql: "SELECT COUNT(*) FROM orders a, orders b", note: "两个订单表不加条件地互相比对，数据量大到算不完，5 秒后会被中止。" }
        ],

        sideMetrics: [
            ["成交额（GMV）", "商品原价加起来"],
            ["净 GMV", "成交额减去退款金额"],
            ["订单数", "按订单号去重，默认只算已送达的订单"],
            ["客单价", "净 GMV 除以已送达订单数"],
            ["新客", "注册不满 7 天的用户"],
            ["复购", "同一个账号在指定时间里下过两单以上"],
            ["核销率", "用掉的券除以领到的券"],
            ["券滥用", "领券后 60 秒内就用掉，而且订单随后退款"],
            ["退款率", "退款订单除以全部订单；商家订单太少时先不计算"],
            ["作弊嫌疑", "一台设备登录 3 个以上账号、券滥用，或者一小时内下 8 单以上"]
        ],

        questions: {
            Q01: "上周（2026-10-26 至 2026-11-01）各城市已送达订单的净 GMV 与订单数，按净 GMV 降序。",
            Q02: "上周（2026-10-26 至 2026-11-01）整体客单价（净 GMV ÷ 已送达订单数），保留两位小数。",
            Q03: "2026 年 10 月每天的订单量，按日期升序。",
            Q04: "上周各城市新客订单占比（new_user=1 的订单数 ÷ 已送达订单数），保留四位小数，按占比降序。",
            Q05: "2026-10-01 至 2026-10-25 注册的用户中，注册后 7 天内下过单的人数与比例（比例保留四位小数）。",
            Q06: "上周下过已送达订单的用户中，复购（≥2 单）人数与复购率（保留四位小数）。",
            Q07: "各优惠券活动的领券数、核销数、核销率（保留四位小数），按核销率降序。",
            Q08: "领券后 60 秒内核销、且该订单已退款的用户数（券滥用嫌疑规模）。",
            Q09: "各活动已核销订单的优惠金额占 GMV 比例（保留四位小数），按比例降序。",
            Q10: "关联账号数 ≥3 的设备及其账号数，按账号数降序、设备 ID 升序。",
            Q11: "设备 D900001 与 D900002 一共关联多少账号，其中 2026-10-01 及以后注册的新账号多少。",
            Q12: "同一下单 IP、同一天出现 ≥5 个不同账号的“IP-日”组合数。",
            Q13: "各城市用户的平均关联设备数（保留两位小数），按均值降序。",
            Q14: "上周各城市骑手人均完成单量（已送达订单数 ÷ 有记录的骑手数，保留两位小数），按人均降序。",
            Q15: "订单量 ≥20 的骑手中，取消率最高的 10 位（取消率保留四位小数，按取消率降序、订单数降序）。",
            Q16: "骑手常用设备同时被某个用户账号使用的“骑手-账号”对数。",
            Q17: "由“与该用户共用设备”的骑手配送的订单数（自交易嫌疑订单）。",
            Q18: "订单量 ≥50 的商家中，退款率最高的 10 家（退款率保留四位小数，按退款率降序、订单数降序）。",
            Q19: "退款金额最高的 5 家商家及其退款金额（保留两位小数）。",
            Q20: "各商家品类的 GMV 与占全部订单 GMV 的比例（比例保留四位小数），按 GMV 降序。",
            Q21: "存在“同一用户、同一小时 ≥8 单”的用户数与涉及订单数。",
            Q22: "退款订单数 ≥3 的用户数。",
            Q23: "三类作弊嫌疑（多账号设备、60 秒券滥用、小时频次 ≥8）合并去重后的用户数。",
            Q24: "设备 D900001/D900002 关联账号在 2026-10-18 至 10-24（处置前）与 10-25 至 10-31（处置后）的订单数与下单用户数对比，按时间顺序输出。"
        },

        schemaRewrites: {
            scenario: [
                { from: "全部为合成数据", to: "全部数据都是模拟生成的" }
            ],
            pitfalls: [
                { from: "净 GMV = SUM(pay_amt - refund_amt)，不是 SUM(gmv)。", to: "净 GMV 要用实付金额减去退款金额，不能直接加商品原价。" },
                { from: "一台设备可对应多个账号；统计账号规模必须按 user_id 去重，不能数设备行。", to: "一台设备可以对应多个账号，所以统计账号规模要按 user_id 去重，不能数设备行。" },
                { from: "活跃用户按去重 user_id，不是订单数；未送达订单是否计入要看问题口径。", to: "活跃用户要按 user_id 去重，不能数订单；未送达的订单算不算，要看题目要求。" },
                { from: "所有数据为合成数据，结论只用于方法演示。", to: "所有数据都是模拟生成的，结论只用来演示方法。" }
            ]
        },

        guard: {
            unclosedComment: "注释没有写完。",
            unclosedBracket: "方括号没有配对。",
            unclosedQuote: "引号没有配对。",
            nullChar: "语句里含有无法识别的空字符。",
            multiStatement: "一次只能提交一条语句，这里实际有 {n} 条。",
            notQuery: "只能执行查询语句，查询要以 SELECT、WITH 或 EXPLAIN 开头。",
            writeKeyword: "这条语句会修改或删除数据（{op}），已经拒绝。"
        },

        compare: {
            columnMismatch: "列数对不上",
            columnDetail: "你返回了 {got} 列，这道题需要 {want} 列。",
            tooManyReason: "结果太多了",
            tooManyDetail: "查询返回的行数超过页面能比较的数量，没法完整比对。",
            rowMismatch: "行数对不上",
            rowDetail: "你返回了 {got} 行，这道题需要 {want} 行。",
            resultMismatch: "结果对不上",
            resultDetail: "结果和标准答案对不上。",
            passReason: "通过",
            passDetail: "列数和每一行的结果都对上了。",
            rowDiff: "第 {n} 行不一样。你的是 {got}，标准答案是 {want}。",
            rowMissing: "你返回的 {row} 这一行不在标准答案里。"
        },

        worker: {
            noData: "示例数据还没有加载。",
            unknownCommand: "未知命令：{cmd}"
        }
    };
}(typeof window !== "undefined" ? window : (typeof self !== "undefined" ? self : this)));
