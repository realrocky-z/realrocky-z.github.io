/* 交互工作台：题库、编辑器、sql.js Worker、判定、审查面板。 */
(function () {
    "use strict";

    var $ = function (sel, root) { return (root || document).querySelector(sel); };
    var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

    var CATEGORY_ORDER = ["规模", "新客", "优惠券", "设备", "骑手", "商家", "作弊特征", "策略评估"];
    var DIFFICULTY_LABEL = { easy: "简单", medium: "中等", hard: "较难" };

    var TOOLS = [
        { name: "get_scenario", desc: "返回业务背景、时间范围、六张表的清单、表之间怎么关联，以及哪里容易算错。" },
        { name: "get_schema", desc: "返回每张表有哪些字段、字段是什么类型、是什么意思，各有多少行。也可以只查一张表。" },
        { name: "search_columns", desc: "不知道某个字段在哪张表里时，可以用关键词搜字段名和说明。" },
        { name: "get_metrics", desc: "返回成交额、退款后的成交额、客单价、新客、复购、优惠券核销、退款和作弊嫌疑这些指标各自怎么算。" },
        { name: "validate_sql", desc: "只检查语句能不能执行，不真的取数。写操作和一次提交多条语句都会被拒绝。" },
        { name: "run_sql", desc: "真正执行查询。默认返回 100 行，最多 200 行，超过 5 秒就中止。" }
    ];

    var EXAMPLES = [
        { title: "写操作", sql: "DROP TABLE orders", note: "这条语句在执行之前就被拦下，不需要加载数据。" },
        { title: "多语句", sql: "SELECT 1; DROP TABLE orders", note: "这里其实写了两条语句，而页面只接受一条。" },
        { title: "WITH + DELETE", sql: "WITH x AS (SELECT 1) DELETE FROM orders", note: "前面加上 WITH，也绕不过对写操作的检查。" },
        { title: "PRAGMA", sql: "PRAGMA table_info(orders)", note: "它既不是查询，也不是查看执行计划，所以不在允许范围内。" },
        { title: "结果太多", sql: "SELECT * FROM orders", note: "页面最多显示 200 行，判定时用前 1000 行。" },
        { title: "算不完", sql: "SELECT COUNT(*) FROM orders a, orders b", note: "两个订单表不加条件地互相比对，数据量大到算不完，5 秒后会被中止。" }
    ];

    var state = {
        manifest: null,
        questions: [],
        byId: new Map(),
        schema: null,
        solutions: null,
        current: null,
        progress: new Set(),
        drafts: {},
        audit: [],
        worker: null,
        buffer: null,
        dbReady: false,
        loading: false,
        typing: null,
        seq: 0,
        pending: new Map()
    };

    document.addEventListener("DOMContentLoaded", init);

    function init() {
        bindTabs();
        bindActions();
        renderExamples();
        renderTools();

        if (location.protocol === "file:") {
            $("#file-banner").hidden = false;
            setDbStatus("file:// 无法加载数据", false);
            $("#q-title").textContent = "请用本地 HTTP 服务打开";
            $("#q-text").textContent = "浏览器不允许直接双击打开时读取页面需要的运行文件和数据。按页面底部的提示启动本地服务，就能正常使用。";
            $("#btn-load").disabled = true;
            return;
        }

        Promise.all([
            fetchJSON("data/manifest.json"),
            fetchJSON("data/questions.json"),
            fetchJSON("data/schema.json")
        ]).then(function (result) {
            state.manifest = result[0];
            state.questions = result[1];
            state.schema = result[2];
            state.questions.forEach(function (q) { state.byId.set(q.id, q); });
            loadProgress();
            loadDrafts();
            renderQuestions();
            renderScenario();
            updateLoadNote();
            selectQuestion(state.questions[0], true);
        }).catch(function (err) {
            $("#q-title").textContent = "数据加载失败";
            $("#q-text").textContent = err.message + "。请确认通过本地 HTTP 服务打开页面。";
        });
    }

    /* ===== Tabs & static panels ===== */

    function bindTabs() {
        $$(".lab-tab").forEach(function (tab) {
            tab.addEventListener("click", function () { switchTab(tab.getAttribute("data-tab")); });
        });
    }

    function switchTab(name) {
        $$(".lab-tab").forEach(function (tab) {
            var on = tab.getAttribute("data-tab") === name;
            tab.classList.toggle("active", on);
            tab.setAttribute("aria-selected", on ? "true" : "false");
        });
        $$(".lab-panel").forEach(function (panel) {
            panel.classList.toggle("active", panel.id === "tab-" + name);
        });
    }

    function renderTools() {
        var box = $("#tools-grid");
        if (!box) return;
        TOOLS.forEach(function (tool) {
            var card = document.createElement("article");
            card.className = "tool-card";
            var name = document.createElement("code");
            name.textContent = tool.name;
            var desc = document.createElement("p");
            desc.textContent = tool.desc;
            card.appendChild(name);
            card.appendChild(desc);
            box.appendChild(card);
        });
    }

    function renderExamples() {
        var box = $("#examples");
        if (!box) return;
        EXAMPLES.forEach(function (item) {
            var btn = document.createElement("button");
            btn.type = "button";
            btn.className = "example-btn";
            var title = document.createElement("strong");
            title.textContent = item.title;
            var code = document.createElement("code");
            code.textContent = item.sql;
            var note = document.createElement("span");
            note.textContent = item.note;
            btn.appendChild(title);
            btn.appendChild(code);
            btn.appendChild(note);
            btn.addEventListener("click", function () { runExample(item.sql); });
            box.appendChild(btn);
        });
    }

    /* ===== Questions ===== */

    function renderQuestions() {
        var list = $("#q-list");
        var select = $("#q-select");
        list.innerHTML = "";
        select.innerHTML = "";

        CATEGORY_ORDER.forEach(function (category) {
            var items = state.questions.filter(function (q) { return q.category === category; });
            if (!items.length) return;

            var heading = document.createElement("div");
            heading.className = "q-group-title";
            heading.textContent = category;
            list.appendChild(heading);

            var group = document.createElement("optgroup");
            group.label = category;

            items.forEach(function (q) {
                var btn = document.createElement("button");
                btn.type = "button";
                btn.className = "q-item";
                btn.setAttribute("data-id", q.id);
                var id = document.createElement("span");
                id.className = "q-id";
                id.textContent = q.id;
                var dot = document.createElement("span");
                dot.className = "q-diff " + q.difficulty;
                dot.title = DIFFICULTY_LABEL[q.difficulty] || q.difficulty;
                btn.appendChild(id);
                btn.appendChild(document.createTextNode(shorten(q.question, 26)));
                btn.appendChild(dot);
                btn.addEventListener("click", function () { selectQuestion(q); });
                list.appendChild(btn);

                var opt = document.createElement("option");
                opt.value = q.id;
                opt.textContent = q.id + " · " + shorten(q.question, 40);
                group.appendChild(opt);
            });
            select.appendChild(group);
        });

        select.addEventListener("change", function () {
            var q = state.byId.get(select.value);
            if (q) selectQuestion(q);
        });
        updateProgressUI();
    }

    function selectQuestion(q, silent) {
        if (!q) return;
        stopTyping(true);
        state.current = q;

        var meta = $("#q-meta");
        meta.innerHTML = "";
        [q.id, q.category, q.skill].forEach(function (text) {
            var tag = document.createElement("span");
            tag.textContent = text;
            meta.appendChild(tag);
        });
        var diff = document.createElement("span");
        diff.className = "diff-" + q.difficulty;
        diff.textContent = DIFFICULTY_LABEL[q.difficulty] || q.difficulty;
        meta.appendChild(diff);

        $("#q-title").textContent = q.category + " · " + q.id;
        $("#q-text").textContent = q.question;

        $("#sql-editor").value = state.drafts[q.id] || "";
        $("#verdict").hidden = true;
        showResultMessage("点「执行」之后，结果会显示在这里。");

        $$(".q-item").forEach(function (btn) {
            btn.classList.toggle("active", btn.getAttribute("data-id") === q.id);
        });
        $("#q-select").value = q.id;

        if (!silent && window.innerWidth <= 820) {
            $("#sql-editor").scrollIntoView({ behavior: "smooth", block: "center" });
        }
    }

    function updateProgressUI() {
        $("#progress-label").textContent = state.progress.size + "/" + state.questions.length;
        $$(".q-item").forEach(function (btn) {
            btn.classList.toggle("done", state.progress.has(btn.getAttribute("data-id")));
        });
    }

    function markDone(id) {
        if (state.progress.has(id)) return;
        state.progress.add(id);
        saveProgress();
        updateProgressUI();
    }

    /* ===== Actions ===== */

    function bindActions() {
        $("#btn-load").addEventListener("click", loadDb);
        $("#btn-validate").addEventListener("click", onValidate);
        $("#btn-run").addEventListener("click", onRun);
        $("#btn-clear").addEventListener("click", function () {
            stopTyping(false);
            $("#sql-editor").value = "";
            saveDraft();
            $("#verdict").hidden = true;
            showResultMessage("执行后会显示结果。");
        });
        $("#btn-solution").addEventListener("click", startReferenceTyping);
        bindDownload();
        $$("[data-goto]").forEach(function (btn) {
            btn.addEventListener("click", function () {
                switchTab(btn.getAttribute("data-goto"));
                window.scrollTo({ top: 0, behavior: "smooth" });
            });
        });
        $("#btn-reset-progress").addEventListener("click", function () {
            if (!window.confirm("要清空本地保存的做题记录吗？")) return;
            state.progress.clear();
            saveProgress();
            updateProgressUI();
        });
        $("#sql-editor").addEventListener("input", function () {
            if (state.typing) stopTyping(false);
            saveDraft();
        });
    }

    function currentSql() {
        return $("#sql-editor").value.trim();
    }

    function onValidate() {
        var sql = currentSql();
        var g = QuagentGuard.guard(sql);
        if (!g.ok) {
            audit("validate", sql, false, g.error);
            showVerdict("fail", "已拦截", g.error);
            showResultMessage("这条语句没有进入数据库。");
            return;
        }
        if (!state.dbReady) {
            showVerdict("warn", "先加载示例数据", "先点「加载示例数据」。你也可以先试一条一定会被拦住的语句。");
            return;
        }
        setBusy(true);
        callWorker("validate", { sql: g.sql }, 5000).then(function (res) {
            showPlan(res.plan, g.sql);
            showVerdict("pass", "检查通过", "语句可以执行。下面显示的是执行计划，还没有真正取数。");
            audit("validate", g.sql, true, "");
        }).catch(function (err) {
            showVerdict("fail", "检查没通过", err.message);
            audit("validate", g.sql, false, err.message);
        }).then(function () { setBusy(false); });
    }

    function onRun() {
        var sql = currentSql();
        var g = QuagentGuard.guard(sql);
        if (!g.ok) {
            audit("run", sql, false, g.error);
            showVerdict("fail", "已拦截", g.error);
            showResultMessage("这条语句没有进入数据库。");
            return;
        }
        if (!state.dbReady) {
            showVerdict("warn", "先加载示例数据", "示例数据压缩后大约 3.5MB，加载以后所有查询都在你自己的浏览器里完成。");
            return;
        }
        setBusy(true);
        callWorker("run", { sql: g.sql, limit: 1000 }, 5000).then(function (res) {
            renderResult(res);
            audit("run", g.sql, true, "", res.elapsed_ms);
            var q = state.current;
            if (!q) return;
            var grade = QuagentCompare.grade(res, q);
            if (grade.pass) {
                markDone(q.id);
                showVerdict("pass", "通过 · " + q.id, grade.detail);
            } else {
                showVerdict("fail", grade.reason, grade.detail);
            }
        }).catch(function (err) {
            if (err.timeout) {
                showVerdict("fail", "查询超时", "这条语句超过 5 秒还没有算完，页面把它停掉了，所以不会卡住。重新加载示例数据后可以继续。");
                audit("run", sql, true, "超过 5 秒，已中止");
                setDbStatus("查询已中止，请重新加载数据", false);
            } else {
                showVerdict("fail", "执行失败", err.message);
                audit("run", g.sql, true, err.message);
            }
        }).then(function () { setBusy(false); });
    }

    function runExample(sql) {
        stopTyping(false);
        switchTab("workbench");
        $("#sql-editor").value = sql;
        saveDraft();
        onRun();
    }

    function startReferenceTyping() {
        if (state.typing) {
            stopTyping(true);
            return;
        }
        if (!state.current) return;
        var id = state.current.id;

        if (state.solutions) {
            beginReference(id);
            return;
        }
        fetchJSON("data/solutions.json").then(function (solutions) {
            state.solutions = solutions;
            if (state.current && state.current.id === id) beginReference(id);
        }).catch(function (err) {
            showVerdict("fail", "参考写法没有加载出来", "原因是 " + err.message + "。");
        });
    }

    function beginReference(id) {
        var sql = (state.solutions && state.solutions[id]) || "";
        if (!sql) {
            showVerdict("fail", "这道题没有参考写法", "换一道题再试。");
            return;
        }
        var editor = $("#sql-editor");
        if (editor.value.trim() && !window.confirm(
            "参考写法是预置的标准答案，不是模型现场生成，会覆盖编辑器内容，继续吗？")) {
            return;
        }
        $("#verdict").hidden = true;
        typeReference(sql);
    }

    function typeReference(sql) {
        var editor = $("#sql-editor");
        var reduced = window.matchMedia &&
            window.matchMedia("(prefers-reduced-motion: reduce)").matches;
        editor.value = "";
        editor.focus();

        if (reduced) {
            editor.value = sql;
            editor.selectionStart = editor.selectionEnd = sql.length;
            editor.scrollTop = editor.scrollHeight;
            saveDraft();
            return;
        }

        setBusy(true);
        $("#btn-solution").textContent = "停止";
        var token = { timer: null };
        state.typing = token;
        var i = 0;

        function step() {
            if (state.typing !== token) return;
            if (i >= sql.length) {
                stopTyping(true);
                return;
            }
            i += 1;
            editor.value = sql.slice(0, i);
            editor.selectionStart = editor.selectionEnd = i;
            editor.scrollTop = editor.scrollHeight;
            var delay = sql.charAt(i - 1) === "\n" ? 60 : 12;
            token.timer = setTimeout(step, delay + Math.random() * 10);
        }
        step();
    }

    function stopTyping(save) {
        var token = state.typing;
        if (!token) return;
        clearTimeout(token.timer);
        state.typing = null;
        var btn = $("#btn-solution");
        if (btn) btn.textContent = "参考写法";
        setBusy(false);
        if (save) saveDraft();
    }

    /* ===== Download ===== */

    var PACKAGE_URL = "../../files/work/quagent-local.zip";

    function bindDownload() {
        var btn = $("#btn-download");
        if (!btn) return;
        btn.addEventListener("click", function () {
            var status = $("#download-status");
            status.hidden = false;
            status.textContent = "正在准备下载。";
            btn.disabled = true;
            btn.textContent = "正在下载 0%";

            fetch(PACKAGE_URL).then(function (res) {
                if (!res.ok) throw new Error("HTTP " + res.status);
                var total = Number(res.headers.get("content-length")) || 0;
                if (!res.body || !res.body.getReader) return res.blob();
                var reader = res.body.getReader();
                var chunks = [];
                var received = 0;

                function pump() {
                    return reader.read().then(function (part) {
                        if (part.done) return null;
                        chunks.push(part.value);
                        received += part.value.length;
                        if (total) {
                            btn.textContent = "正在下载 " +
                                Math.min(99, Math.round(received / total * 100)) + "%";
                        } else {
                            btn.textContent = "正在下载 " +
                                (received / 1048576).toFixed(1) + "MB";
                        }
                        return pump();
                    });
                }

                return pump().then(function () {
                    return new Blob(chunks, { type: "application/zip" });
                });
            }).then(function (blob) {
                var url = URL.createObjectURL(blob);
                var a = document.createElement("a");
                a.href = url;
                a.download = "quagent-local.zip";
                document.body.appendChild(a);
                a.click();
                a.remove();
                setTimeout(function () { URL.revokeObjectURL(url); }, 10000);
                btn.textContent = "重新下载";
                btn.disabled = false;
                status.textContent = "下载完成，解压后按下面的步骤运行。";
            }).catch(function (err) {
                btn.textContent = "重试下载";
                btn.disabled = false;
                status.textContent = "下载没有成功，原因是 " + err.message +
                    "。可以用下面的直接下载链接。";
            });
        });
    }

    /* ===== Worker & database ===== */

    function loadDb() {
        if (state.loading) return;
        state.loading = true;
        var btn = $("#btn-load");
        btn.disabled = true;

        var ready = state.buffer ? Promise.resolve(state.buffer) : downloadDb(btn);
        ready.then(function (buffer) {
            state.buffer = buffer;
            btn.textContent = "正在准备";
            return initWorkerFromBuffer();
        }).then(function () {
            btn.textContent = "重新加载示例数据";
            btn.disabled = false;
            state.loading = false;
        }).catch(function (err) {
            setDbStatus("加载失败，原因是 " + err.message, false);
            btn.textContent = "重试";
            btn.disabled = false;
            state.loading = false;
        });
    }

    function downloadDb(btn) {
        btn.textContent = "正在下载 0%";
        return fetch("data/anticheat-demo.db.gz").then(function (res) {
            if (!res.ok) throw new Error("HTTP " + res.status);
            var total = Number(res.headers.get("content-length")) || state.manifest.db_gz_bytes;
            if (!res.body || !res.body.getReader) return res.arrayBuffer().then(gunzip);
            var reader = res.body.getReader();
            var chunks = [];
            var received = 0;
            function pump() {
                return reader.read().then(function (part) {
                    if (part.done) return null;
                    chunks.push(part.value);
                    received += part.value.length;
                    btn.textContent = "正在下载 " + Math.min(99, Math.round(received / total * 100)) + "%";
                    return pump();
                });
            }
            return pump().then(function () {
                var gz = new Uint8Array(received);
                var offset = 0;
                chunks.forEach(function (chunk) { gz.set(chunk, offset); offset += chunk.length; });
                btn.textContent = "正在解压";
                return gunzip(gz);
            });
        });
    }

    function gunzip(bytes) {
        if (typeof DecompressionStream === "function") {
            var stream = new Blob([bytes]).stream().pipeThrough(new DecompressionStream("gzip"));
            return new Response(stream).arrayBuffer();
        }
        if (window.fflate && window.fflate.gunzipSync) {
            return Promise.resolve(window.fflate.gunzipSync(bytes).buffer);
        }
        return Promise.reject(new Error("浏览器不支持 gzip 解压，请更新浏览器"));
    }

    function initWorkerFromBuffer() {
        ensureWorker();
        return callWorker("init", { buffer: state.buffer }, 20000).then(function () {
            state.dbReady = true;
            setDbStatus("示例数据已经准备好", true);
        });
    }

    function ensureWorker() {
        if (state.worker) return;
        state.worker = new Worker("js/worker.js");
        state.worker.onmessage = function (event) {
            var msg = event.data;
            var pending = state.pending.get(msg.id);
            if (!pending) return;
            clearTimeout(pending.timer);
            state.pending.delete(msg.id);
            if (msg.ok) pending.resolve(msg.payload);
            else pending.reject(new Error(msg.payload && msg.payload.error ? msg.payload.error : "未知错误"));
        };
        state.worker.onerror = function (event) {
            failAll(new Error(event.message || "Worker 错误"));
        };
    }

    function callWorker(cmd, payload, timeoutMs) {
        ensureWorker();
        return new Promise(function (resolve, reject) {
            var id = ++state.seq;
            var timer = setTimeout(function () {
                state.pending.delete(id);
                killWorker();
                var err = new Error("超过 " + (timeoutMs / 1000) + " 秒未完成");
                err.timeout = true;
                reject(err);
            }, timeoutMs);
            state.pending.set(id, { resolve: resolve, reject: reject, timer: timer });
            var msg = { id: id, cmd: cmd };
            Object.keys(payload).forEach(function (key) { msg[key] = payload[key]; });
            state.worker.postMessage(msg);
        });
    }

    function killWorker() {
        if (state.worker) {
            state.worker.terminate();
            state.worker = null;
        }
        state.dbReady = false;
        failAll(new Error("Worker 已终止"));
        setDbStatus("已终止，请重新加载", false);
    }

    function failAll(err) {
        state.pending.forEach(function (pending) {
            clearTimeout(pending.timer);
            pending.reject(err);
        });
        state.pending.clear();
    }

    function setBusy(on) {
        $("#btn-validate").disabled = on;
        $("#btn-run").disabled = on;
    }

    function setDbStatus(text, ready) {
        var el = $("#db-status");
        el.textContent = text;
        el.classList.toggle("ready", Boolean(ready));
    }

    /* ===== Rendering ===== */

    function showVerdict(kind, title, detail) {
        var box = $("#verdict");
        box.className = "verdict " + kind;
        box.innerHTML = "";
        var strong = document.createElement("strong");
        strong.textContent = title;
        var body = document.createElement("span");
        body.textContent = detail;
        box.appendChild(strong);
        box.appendChild(body);
        box.hidden = false;
    }

    function showResultMessage(text) {
        var wrap = $("#result-wrap");
        wrap.innerHTML = "";
        var p = document.createElement("p");
        p.className = "empty";
        p.textContent = text;
        wrap.appendChild(p);
    }

    function renderResult(res) {
        var wrap = $("#result-wrap");
        wrap.innerHTML = "";

        var meta = document.createElement("div");
        meta.className = "res-meta";
        meta.appendChild(metaItem("共", res.row_count + " 行" +
            (res.truncated ? "，页面只显示前 200 行" : "")));
        meta.appendChild(metaItem("用了", res.elapsed_ms + " 毫秒"));
        meta.appendChild(metaItem("返回", res.columns.length + " 列"));
        wrap.appendChild(meta);

        if (!res.rows.length) {
            var empty = document.createElement("p");
            empty.className = "empty";
            empty.textContent = "查询成功，但没有返回行。";
            wrap.appendChild(empty);
            return;
        }

        var table = document.createElement("table");
        table.className = "res-table";
        var thead = document.createElement("thead");
        var headRow = document.createElement("tr");
        res.columns.forEach(function (name) {
            var th = document.createElement("th");
            th.textContent = name;
            headRow.appendChild(th);
        });
        thead.appendChild(headRow);
        table.appendChild(thead);

        var tbody = document.createElement("tbody");
        res.rows.slice(0, 200).forEach(function (row) {
            var tr = document.createElement("tr");
            row.forEach(function (value) {
                var td = document.createElement("td");
                if (value === null) {
                    td.textContent = "NULL";
                    td.className = "null";
                } else {
                    td.textContent = String(value);
                }
                tr.appendChild(td);
            });
            tbody.appendChild(tr);
        });
        table.appendChild(tbody);
        wrap.appendChild(table);

        if (res.rows.length > 200) {
            var note = document.createElement("p");
            note.className = "empty";
            note.textContent = "页面最多显示 200 行，判定时用的是前 1000 行。";
            wrap.appendChild(note);
        }
    }

    function metaItem(label, value) {
        var span = document.createElement("span");
        var b = document.createElement("b");
        b.textContent = label + " ";
        span.appendChild(b);
        span.appendChild(document.createTextNode(String(value)));
        return span;
    }

    function showPlan(plan, sql) {
        var wrap = $("#result-wrap");
        wrap.innerHTML = "";
        var label = document.createElement("p");
        label.className = "empty";
        label.textContent = "执行计划（还没有真正取数）";
        wrap.appendChild(label);
        var pre = document.createElement("pre");
        pre.className = "plan";
        if (!plan || !plan.length) {
            pre.textContent = "语句本身没问题，但数据库没有给出执行计划。";
        } else {
            pre.textContent = plan.map(function (row) {
                return String(row[row.length - 1]);
            }).join("\n");
        }
        wrap.appendChild(pre);
        var sqlNote = document.createElement("p");
        sqlNote.className = "empty";
        sqlNote.textContent = "这条语句是 " + shorten(sql, 160) + "。";
        wrap.appendChild(sqlNote);
    }

    function renderScenario() {
        var s = state.schema;
        var box = $("#schema-grid");
        var counts = s.demo ? s.demo.row_counts : {};
        ["users", "devices", "riders", "merchants", "orders", "coupons"].forEach(function (name) {
            var meta = s.tables[name];
            if (!meta) return;
            var card = document.createElement("article");
            card.className = "schema-card";

            var count = document.createElement("span");
            count.className = "count";
            count.textContent = (counts[name] || 0).toLocaleString() + " 行";
            var title = document.createElement("h3");
            title.textContent = name;
            card.appendChild(count);
            card.appendChild(title);

            var grain = document.createElement("p");
            grain.className = "grain";
            grain.textContent = meta.grain;
            card.appendChild(grain);

            var table = document.createElement("table");
            table.className = "col-table";
            Object.keys(meta.columns).forEach(function (col) {
                var info = meta.columns[col];
                var tr = document.createElement("tr");
                var nameCell = document.createElement("td");
                nameCell.textContent = col;
                var descCell = document.createElement("td");
                descCell.textContent = info.desc + "（" + info.type + "）";
                tr.appendChild(nameCell);
                tr.appendChild(descCell);
                table.appendChild(tr);
            });
            card.appendChild(table);
            box.appendChild(card);
        });

        var joins = $("#joins-box");
        joins.innerHTML = "";
        if (s.scenario) {
            var p = document.createElement("p");
            p.textContent = s.scenario.business + "数据的时间范围是 " +
                s.scenario.data_range.join(" 到 ") + "，时区是 " + s.scenario.timezone +
                "，金额按" + s.scenario.currency + "计算。";
            joins.appendChild(p);
            var disc = document.createElement("p");
            disc.textContent = plainScenarioText(s.scenario.disclaimer);
            joins.appendChild(disc);
        }
        appendList(joins, "表之间怎么关联", (s.join_keys || []).map(plainJoin));
        appendList(joins, "容易算错的地方", (s.pitfalls || []).map(plainPitfall));
        if (s.demo) {
            var note = document.createElement("p");
            note.textContent = "为了减小下载体积，示例数据里没有 events 表。24 道练习题都用不到它，其余六张表和原项目一致。";
            joins.appendChild(note);
        }

        renderSideReference();

        fetch("data/metrics.html").then(function (res) {
            if (!res.ok) throw new Error("HTTP " + res.status);
            return res.text();
        }).then(function (text) {
            $("#metrics-box").innerHTML = text;
        }).catch(function (err) {
            $("#metrics-box").textContent = "指标说明没有加载出来，原因是 " + err.message + "。";
        });

        updateLoadNote();
    }

    function renderSideReference() {
        var s = state.schema;
        if (!s) return;
        var counts = s.demo ? s.demo.row_counts : {};

        var tables = $("#side-tables");
        tables.innerHTML = "";
        ["users", "devices", "riders", "merchants", "orders", "coupons"].forEach(function (name) {
            var meta = s.tables[name];
            if (!meta) return;
            var row = document.createElement("div");
            row.className = "side-table-row";
            var title = document.createElement("code");
            title.textContent = name;
            var desc = document.createElement("span");
            desc.textContent = meta.grain + "（" + (counts[name] || 0).toLocaleString() + " 行）";
            row.appendChild(title);
            row.appendChild(desc);
            tables.appendChild(row);
        });

        var joins = $("#side-joins");
        joins.innerHTML = "";
        (s.join_keys || []).forEach(function (key) {
            var li = document.createElement("li");
            li.textContent = plainJoin(key);
            joins.appendChild(li);
        });

        var metrics = [
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
        ];
        var box = $("#side-metrics");
        box.innerHTML = "";
        metrics.forEach(function (item) {
            var row = document.createElement("div");
            row.className = "metric-row";
            var label = document.createElement("b");
            label.textContent = item[0];
            var text = document.createElement("span");
            text.textContent = item[1];
            row.appendChild(label);
            row.appendChild(text);
            box.appendChild(row);
        });
    }

    function appendList(parent, heading, items) {
        if (!items || !items.length) return;
        var h = document.createElement("h3");
        h.textContent = heading;
        parent.appendChild(h);
        var ul = document.createElement("ul");
        items.forEach(function (item) {
            var li = document.createElement("li");
            li.textContent = item;
            ul.appendChild(li);
        });
        parent.appendChild(ul);
    }

    function updateLoadNote() {
        var note = $("#load-note");
        if (note && state.manifest) {
            note.textContent = "六张表，其中 60,430 个订单、20,150 张优惠券，压缩后大约 " +
                (state.manifest.db_gz_bytes / 1048576).toFixed(1) +
                "MB。第一次点加载会下载这份数据，之后查询都在浏览器里完成，不会发到任何服务器。";
        }
    }

    /* ===== Audit ===== */

    function audit(cmd, sql, allowed, reason, ms) {
        state.audit.unshift({
            time: new Date().toTimeString().slice(0, 8),
            cmd: cmd,
            sql: String(sql || "").replace(/\s+/g, " ").slice(0, 160),
            allowed: Boolean(allowed),
            reason: reason || "",
            ms: typeof ms === "number" ? ms : null
        });
        if (state.audit.length > 30) state.audit.pop();
        renderMiniLog();
        renderAudit();
    }

    function renderMiniLog() {
        var box = $("#mini-log");
        if (!box) return;
        box.innerHTML = "";
        if (!state.audit.length) {
            box.innerHTML = '<p class="empty">还没有操作。</p>';
            return;
        }
        state.audit.slice(0, 5).forEach(function (entry) {
            var row = document.createElement("div");
            row.className = "log-row";
            var flag = document.createElement("span");
            flag.className = "log-flag " + (entry.allowed ? "ok" : "no");
            flag.textContent = entry.allowed ? "放行" : "拒绝";
            var text = document.createElement("span");
            text.className = "log-sql";
            text.textContent = entry.sql;
            text.title = (entry.reason || entry.sql);
            row.appendChild(flag);
            row.appendChild(text);
            box.appendChild(row);
        });
    }

    function renderAudit() {
        var box = $("#audit-box");
        box.innerHTML = "";
        if (!state.audit.length) {
            box.innerHTML = '<p class="empty">还没有操作。</p>';
            return;
        }
        state.audit.forEach(function (entry) {
            var row = document.createElement("div");
            row.className = "audit-row";

            var time = document.createElement("span");
            time.textContent = entry.time;

            var flag = document.createElement("span");
            flag.className = entry.allowed ? "ok" : "no";
            flag.textContent = entry.allowed ? "放行" : "拒绝";

            var sql = document.createElement("span");
            sql.className = "sql";
            sql.textContent = entry.sql;
            sql.title = entry.sql;

            var reason = document.createElement("span");
            reason.textContent = entry.reason || (entry.ms !== null ? entry.ms + " ms" : "");

            row.appendChild(time);
            row.appendChild(flag);
            row.appendChild(sql);
            row.appendChild(reason);
            box.appendChild(row);
        });
    }

    /* ===== Storage ===== */

    function storageKey(prefix) {
        return prefix + (state.manifest ? state.manifest.questions_sha256 : "dev");
    }

    function loadProgress() {
        try {
            var raw = window.localStorage.getItem(storageKey("quagent-progress:"));
            state.progress = new Set(JSON.parse(raw || "[]"));
        } catch (err) {
            state.progress = new Set();
        }
    }

    function saveProgress() {
        try {
            window.localStorage.setItem(storageKey("quagent-progress:"),
                JSON.stringify(Array.from(state.progress)));
        } catch (err) { /* 隐私模式或配额不足时忽略 */ }
    }

    function loadDrafts() {
        try {
            var raw = window.localStorage.getItem(storageKey("quagent-drafts:"));
            state.drafts = JSON.parse(raw || "{}") || {};
        } catch (err) {
            state.drafts = {};
        }
    }

    function saveDraft() {
        if (!state.current) return;
        state.drafts[state.current.id] = $("#sql-editor").value;
        try {
            window.localStorage.setItem(storageKey("quagent-drafts:"),
                JSON.stringify(state.drafts));
        } catch (err) { /* 忽略 */ }
    }

    /* ===== Utilities ===== */

    function fetchJSON(url) {
        return fetch(url).then(function (res) {
            if (!res.ok) throw new Error(url + " HTTP " + res.status);
            return res.json();
        });
    }

    function shorten(text, max) {
        var s = String(text || "");
        return s.length > max ? s.slice(0, max - 1) + "…" : s;
    }

    var TABLE_LABEL = {
        users: "账号", devices: "设备", riders: "骑手",
        merchants: "商家", orders: "订单", coupons: "优惠券"
    };

    function plainJoin(key) {
        var m = String(key).match(/^(\w+)\.(\w+) = (\w+)\.(\w+)$/);
        if (!m) return String(key);
        return (TABLE_LABEL[m[1]] || m[1]) + "表的 " + m[2] +
            " 对应" + (TABLE_LABEL[m[3]] || m[3]) + "表的 " + m[4];
    }

    function plainScenarioText(text) {
        return String(text).replace("全部为合成数据", "全部数据都是模拟生成的");
    }

    function plainPitfall(text) {
        return plainScenarioText(text)
            .replace("净 GMV = SUM(pay_amt - refund_amt)，不是 SUM(gmv)。",
                "净 GMV 要用实付金额减去退款金额，不能直接加商品原价。")
            .replace("一台设备可对应多个账号；统计账号规模必须按 user_id 去重，不能数设备行。",
                "一台设备可以对应多个账号，所以统计账号规模要按 user_id 去重，不能数设备行。")
            .replace("活跃用户按去重 user_id，不是订单数；未送达订单是否计入要看问题口径。",
                "活跃用户要按 user_id 去重，不能数订单；未送达的订单算不算，要看题目要求。")
            .replace("所有数据为合成数据，结论只用于方法演示。",
                "所有数据都是模拟生成的，结论只用来演示方法。");
    }
}());
