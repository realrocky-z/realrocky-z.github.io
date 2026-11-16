/* 交互工作台：题库、编辑器、sql.js Worker、判定、审查面板。 */
(function () {
    "use strict";

    var $ = function (sel, root) { return (root || document).querySelector(sel); };
    var $$ = function (sel, root) { return Array.prototype.slice.call((root || document).querySelectorAll(sel)); };

    var COPY = window.QUAGENT_COPY || {};
    var CATEGORY_ORDER = ["规模", "新客", "优惠券", "设备", "骑手", "商家", "作弊特征", "策略评估"];

    function copyAt(key) {
        return key.split(".").reduce(function (node, part) {
            return node == null ? undefined : node[part];
        }, COPY);
    }

    function fmt(template, values) {
        return String(template == null ? "" : template).replace(/\{(\w+)\}/g, function (_, key) {
            return values[key] == null ? "" : values[key];
        });
    }

    function applyCopy() {
        document.title = copyAt("meta.title") || document.title;
        $$("[data-copy]").forEach(function (el) {
            var text = copyAt(el.getAttribute("data-copy"));
            if (text != null) el.textContent = text;
        });
        $$("[data-copy-html]").forEach(function (el) {
            var html = copyAt(el.getAttribute("data-copy-html"));
            if (html != null) el.innerHTML = html;
        });
        $$("[data-copy-placeholder]").forEach(function (el) {
            var text = copyAt(el.getAttribute("data-copy-placeholder"));
            if (text != null) el.setAttribute("placeholder", text);
        });
        $$("[data-copy-aria]").forEach(function (el) {
            var text = copyAt(el.getAttribute("data-copy-aria"));
            if (text != null) el.setAttribute("aria-label", text);
        });
    }

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
        applyCopy();
        bindTabs();
        bindActions();
        bindModal();
        renderHeroSteps();
        renderSafetyLayers();
        renderExamples();
        renderTools();

        if (location.protocol === "file:") {
            $("#file-banner").hidden = false;
            setDbStatus(copyAt("messages.fileStatus"), false);
            $("#q-title").textContent = copyAt("messages.fileTitle");
            $("#q-text").textContent = copyAt("messages.fileText");
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
            $("#q-title").textContent = copyAt("messages.dataFailTitle");
            $("#q-text").textContent = fmt(copyAt("messages.dataFail"), { reason: err.message });
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

    function renderHeroSteps() {
        var box = $("#hero-steps");
        if (!box) return;
        (COPY.hero && COPY.hero.steps ? COPY.hero.steps : []).forEach(function (html) {
            var li = document.createElement("li");
            li.innerHTML = html;
            box.appendChild(li);
        });
    }

    function renderSafetyLayers() {
        var box = $("#layer-grid");
        if (!box) return;
        (COPY.safety && COPY.safety.layers ? COPY.safety.layers : []).forEach(function (layer) {
            var card = document.createElement("article");
            card.className = "layer-card";
            var no = document.createElement("span");
            no.className = "layer-no";
            no.textContent = layer.no;
            var title = document.createElement("h3");
            title.textContent = layer.title;
            var body = document.createElement("p");
            body.innerHTML = layer.bodyHtml;
            var evidence = document.createElement("p");
            evidence.className = "layer-evidence";
            evidence.innerHTML = layer.evidenceHtml;
            card.appendChild(no);
            card.appendChild(title);
            card.appendChild(body);
            card.appendChild(evidence);
            box.appendChild(card);
        });
    }

    function renderTools() {
        var box = $("#tools-grid");
        if (!box) return;
        (COPY.tools || []).forEach(function (tool) {
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
        (COPY.examples || []).forEach(function (item) {
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

    function questionText(q) {
        return (COPY.questions && COPY.questions[q.id]) || "";
    }

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
                dot.title = (COPY.labels.difficulty || {})[q.difficulty] || q.difficulty;
                btn.appendChild(id);
                btn.appendChild(document.createTextNode(shorten(questionText(q), 26)));
                btn.appendChild(dot);
                btn.addEventListener("click", function () { selectQuestion(q); });
                list.appendChild(btn);

                var opt = document.createElement("option");
                opt.value = q.id;
                opt.textContent = fmt(copyAt("messages.questionOption"),
                    { id: q.id, text: shorten(questionText(q), 40) });
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
        diff.textContent = (COPY.labels.difficulty || {})[q.difficulty] || q.difficulty;
        meta.appendChild(diff);

        $("#q-title").textContent = fmt(copyAt("messages.questionTitle"),
            { category: q.category, id: q.id });
        $("#q-text").textContent = questionText(q);

        $("#sql-editor").value = state.drafts[q.id] || "";
        $("#verdict").hidden = true;
        showResultMessage(copyAt("workbench.resultEmpty"));

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
            showResultMessage(copyAt("workbench.clearResultEmpty"));
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
            openModal(copyAt("modal.reset.title"), copyAt("modal.reset.body"), [
                { label: copyAt("modal.reset.clear"), primary: true, onClick: function () {
                    closeModal();
                    state.progress.clear();
                    saveProgress();
                    updateProgressUI();
                } },
                { label: copyAt("modal.reset.cancel"), onClick: closeModal }
            ]);
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
            showVerdict("fail", copyAt("messages.blockedTitle"), g.error);
            showResultMessage(copyAt("messages.blockedDetail"));
            return;
        }
        if (!state.dbReady) {
            showVerdict("warn", copyAt("messages.validateNoDbTitle"), copyAt("messages.validateNoDbDetail"));
            return;
        }
        setBusy(true);
        callWorker("validate", { sql: g.sql }, 5000).then(function (res) {
            showPlan(res.plan, g.sql);
            showVerdict("pass", copyAt("messages.validatePassTitle"), copyAt("messages.validatePassDetail"));
            audit("validate", g.sql, true, "");
        }).catch(function (err) {
            showVerdict("fail", copyAt("messages.validateFailTitle"), err.message);
            audit("validate", g.sql, false, err.message);
        }).then(function () { setBusy(false); });
    }

    function onRun() {
        var sql = currentSql();
        var g = QuagentGuard.guard(sql);
        if (!g.ok) {
            audit("run", sql, false, g.error);
            showVerdict("fail", copyAt("messages.blockedTitle"), g.error);
            showResultMessage(copyAt("messages.blockedDetail"));
            return;
        }
        if (!state.dbReady) {
            showVerdict("warn", copyAt("messages.validateNoDbTitle"), copyAt("messages.runNoDbDetail"));
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
                showVerdict("pass", fmt(copyAt("messages.passTitle"), { id: q.id }), grade.detail);
            } else {
                showVerdict("fail", grade.reason, grade.detail);
            }
        }).catch(function (err) {
            if (err.timeout) {
                showVerdict("fail", copyAt("messages.timeoutTitle"), copyAt("messages.timeoutDetail"));
                audit("run", sql, true, copyAt("messages.timeoutAudit"));
                setDbStatus(copyAt("messages.timeoutStatus"), false);
            } else {
                showVerdict("fail", copyAt("messages.runFailTitle"), err.message);
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

        openModal(copyAt("modal.choice.title"), copyAt("modal.choice.body"), [
            { label: copyAt("modal.choice.builtin"), primary: true, onClick: function () {
                closeModal();
                showUsageNotice(function () { loadReference(id); });
            } },
            { label: copyAt("modal.choice.apiKey"), onClick: function () {
                closeModal();
                showUsageNotice(null);
            } }
        ]);
    }

    function showUsageNotice(next) {
        var actions = [];
        if (next) {
            actions.push({ label: copyAt("modal.notice.continue"), primary: true, onClick: function () {
                closeModal();
                next();
            } });
        }
        actions.push({ label: copyAt("modal.notice.download"), primary: !next, onClick: function () {
            closeModal();
            scrollToDownload();
        } });
        if (!next) {
            actions.push({ label: copyAt("modal.notice.ok"), onClick: closeModal });
        }
        openModal(copyAt("modal.notice.title"), copyAt("modal.notice.body"), actions);
    }

    function scrollToDownload() {
        var block = $(".download-block");
        if (!block) return;
        if (window.innerWidth <= 820) switchTab("workbench");
        block.scrollIntoView({ behavior: "smooth", block: "center" });
        var btn = $("#btn-download");
        if (btn && !btn.disabled) btn.click();
    }

    function loadReference(id) {
        if (state.solutions) {
            beginReference(id);
            return;
        }
        fetchJSON("data/solutions.json").then(function (solutions) {
            state.solutions = solutions;
            beginReference(id);
        }).catch(function (err) {
            showVerdict("fail", copyAt("reference.loadFailTitle"),
                fmt(copyAt("reference.loadFail"), { reason: err.message }));
        });
    }

    function beginReference(id) {
        if (!state.current || state.current.id !== id) return;
        var sql = (state.solutions && state.solutions[id]) || "";
        if (!sql) {
            showVerdict("fail", copyAt("reference.noSolutionTitle"), copyAt("reference.noSolutionDetail"));
            return;
        }
        var editor = $("#sql-editor");
        if (editor.value.trim()) {
            openModal(copyAt("modal.overwrite.title"), copyAt("modal.overwrite.body"), [
                { label: copyAt("modal.overwrite.fill"), primary: true, onClick: function () {
                    closeModal();
                    $("#verdict").hidden = true;
                    typeReference(sql);
                } },
                { label: copyAt("modal.overwrite.cancel"), onClick: closeModal }
            ]);
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
        $("#btn-solution").textContent = copyAt("reference.stop");
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
        if (btn) btn.textContent = copyAt("workbench.buttons.solution");
        setBusy(false);
        if (save) saveDraft();
    }

    /* ===== Modal ===== */

    var modalReturnFocus = null;
    var modalScroll = "";

    function openModal(title, body, actions) {
        var backdrop = $("#modal");
        if (!backdrop) return;
        modalReturnFocus = document.activeElement;
        modalScroll = document.body.style.overflow;
        document.body.style.overflow = "hidden";
        $("#modal-title").textContent = title;
        $("#modal-body").textContent = body;
        var box = $("#modal-actions");
        box.innerHTML = "";
        actions.forEach(function (action) {
            var btn = document.createElement("button");
            btn.type = "button";
            btn.className = "lab-btn" + (action.primary ? " primary" : "");
            btn.textContent = action.label;
            btn.addEventListener("click", action.onClick);
            box.appendChild(btn);
        });
        backdrop.hidden = false;
        var first = box.querySelector("button");
        if (first) first.focus();
    }

    function closeModal() {
        var backdrop = $("#modal");
        if (!backdrop || backdrop.hidden) return;
        backdrop.hidden = true;
        document.body.style.overflow = modalScroll;
        if (modalReturnFocus && typeof modalReturnFocus.focus === "function") {
            try { modalReturnFocus.focus(); } catch (err) { /* 元素可能已被移除 */ }
        }
        modalReturnFocus = null;
    }

    function bindModal() {
        var backdrop = $("#modal");
        if (!backdrop) return;
        backdrop.addEventListener("click", function (event) {
            if (event.target === backdrop) closeModal();
        });
        document.addEventListener("keydown", function (event) {
            if (backdrop.hidden) return;
            if (event.key === "Escape") {
                closeModal();
                return;
            }
            if (event.key !== "Tab") return;
            var items = Array.prototype.slice.call(backdrop.querySelectorAll("button"));
            if (!items.length) return;
            var first = items[0];
            var last = items[items.length - 1];
            if (event.shiftKey && document.activeElement === first) {
                event.preventDefault();
                last.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault();
                first.focus();
            }
        });
    }

    /* ===== Download ===== */

    var PACKAGE_URL = "../../files/work/quagent-local.zip";

    function bindDownload() {
        var btn = $("#btn-download");
        if (!btn) return;
        btn.addEventListener("click", function () {
            var status = $("#download-status");
            status.hidden = false;
            status.textContent = copyAt("messages.downloadPreparing");
            btn.disabled = true;
            btn.textContent = fmt(copyAt("messages.downloadProgress"), { pct: 0 });

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
                            btn.textContent = fmt(copyAt("messages.downloadProgress"), {
                                pct: Math.min(99, Math.round(received / total * 100))
                            });
                        } else {
                            btn.textContent = fmt(copyAt("messages.downloadProgressMb"), {
                                mb: (received / 1048576).toFixed(1)
                            });
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
                btn.textContent = copyAt("messages.downloadAgain");
                btn.disabled = false;
                status.textContent = copyAt("messages.downloadDone");
            }).catch(function (err) {
                btn.textContent = copyAt("messages.downloadRetry");
                btn.disabled = false;
                status.textContent = fmt(copyAt("messages.downloadFail"), { reason: err.message });
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
            btn.textContent = copyAt("messages.dbPreparing");
            return initWorkerFromBuffer();
        }).then(function () {
            btn.textContent = copyAt("messages.dbReload");
            btn.disabled = false;
            state.loading = false;
        }).catch(function (err) {
            setDbStatus(fmt(copyAt("messages.dbLoadFail"), { reason: err.message }), false);
            btn.textContent = copyAt("messages.dbRetry");
            btn.disabled = false;
            state.loading = false;
        });
    }

    function downloadDb(btn) {
        btn.textContent = fmt(copyAt("messages.dbDownloading"), { pct: 0 });
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
                    btn.textContent = fmt(copyAt("messages.dbDownloading"), {
                        pct: Math.min(99, Math.round(received / total * 100))
                    });
                    return pump();
                });
            }
            return pump().then(function () {
                var gz = new Uint8Array(received);
                var offset = 0;
                chunks.forEach(function (chunk) { gz.set(chunk, offset); offset += chunk.length; });
                btn.textContent = copyAt("messages.dbUnzip");
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
        return Promise.reject(new Error(copyAt("messages.dbGzip")));
    }

    function initWorkerFromBuffer() {
        ensureWorker();
        return callWorker("init", { buffer: state.buffer }, 20000).then(function () {
            state.dbReady = true;
            setDbStatus(copyAt("messages.dbReady"), true);
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
            else pending.reject(new Error(msg.payload && msg.payload.error ? msg.payload.error : copyAt("messages.workerUnknown")));
        };
        state.worker.onerror = function (event) {
            failAll(new Error(event.message || copyAt("messages.workerError")));
        };
    }

    function callWorker(cmd, payload, timeoutMs) {
        ensureWorker();
        return new Promise(function (resolve, reject) {
            var id = ++state.seq;
            var timer = setTimeout(function () {
                state.pending.delete(id);
                killWorker();
                var err = new Error(fmt(copyAt("messages.workerTimeout"), { seconds: timeoutMs / 1000 }));
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
        failAll(new Error(copyAt("messages.workerKilled")));
        setDbStatus(copyAt("messages.dbTerminated"), false);
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
        meta.appendChild(metaItem(copyAt("messages.resultRowsLabel"),
            fmt(copyAt(res.truncated ? "messages.resultRowsTruncatedValue" : "messages.resultRowsValue"),
                { n: res.row_count })));
        meta.appendChild(metaItem(copyAt("messages.resultElapsedLabel"),
            fmt(copyAt("messages.resultElapsedValue"), { ms: res.elapsed_ms })));
        meta.appendChild(metaItem(copyAt("messages.resultColumnsLabel"),
            fmt(copyAt("messages.resultColumnsValue"), { n: res.columns.length })));
        wrap.appendChild(meta);

        if (!res.rows.length) {
            var empty = document.createElement("p");
            empty.className = "empty";
            empty.textContent = copyAt("messages.resultEmpty");
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
            note.textContent = copyAt("messages.resultLimitNote");
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
        label.textContent = copyAt("messages.planLabel");
        wrap.appendChild(label);
        var pre = document.createElement("pre");
        pre.className = "plan";
        if (!plan || !plan.length) {
            pre.textContent = copyAt("messages.planEmpty");
        } else {
            pre.textContent = plan.map(function (row) {
                return String(row[row.length - 1]);
            }).join("\n");
        }
        wrap.appendChild(pre);
        var sqlNote = document.createElement("p");
        sqlNote.className = "empty";
        sqlNote.textContent = fmt(copyAt("messages.planSql"), { sql: shorten(sql, 160) });
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
            count.textContent = fmt(copyAt("messages.sideRowCount"),
                { n: (counts[name] || 0).toLocaleString() });
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
                descCell.textContent = fmt(copyAt("messages.columnDesc"),
                    { desc: info.desc, type: info.type });
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
            p.textContent = fmt(copyAt("messages.scenarioSentence"), {
                business: s.scenario.business,
                start: s.scenario.data_range[0],
                end: s.scenario.data_range[1],
                tz: s.scenario.timezone,
                currency: s.scenario.currency
            });
            joins.appendChild(p);
            var disc = document.createElement("p");
            disc.textContent = plainScenarioText(s.scenario.disclaimer);
            joins.appendChild(disc);
        }
        appendList(joins, copyAt("scenario.generated.joinsHeading"), (s.join_keys || []).map(plainJoin));
        appendList(joins, copyAt("scenario.generated.pitfallsHeading"), (s.pitfalls || []).map(plainPitfall));
        if (s.demo) {
            var note = document.createElement("p");
            note.textContent = copyAt("scenario.generated.demoNote");
            joins.appendChild(note);
        }

        renderSideReference();

        $("#metrics-box").innerHTML = copyAt("scenario.metricsHtml") || "";

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
            desc.textContent = fmt(copyAt("messages.sideTableDesc"), {
                grain: meta.grain,
                count: (counts[name] || 0).toLocaleString()
            });
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

        var box = $("#side-metrics");
        box.innerHTML = "";
        (COPY.sideMetrics || []).forEach(function (item) {
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
            note.textContent = fmt(copyAt("messages.loadNote"), {
                mb: (state.manifest.db_gz_bytes / 1048576).toFixed(1)
            });
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
            box.innerHTML = '<p class="empty"></p>';
            box.firstChild.textContent = copyAt("safety.audit.empty");
            return;
        }
        state.audit.slice(0, 5).forEach(function (entry) {
            var row = document.createElement("div");
            row.className = "log-row";
            var flag = document.createElement("span");
            flag.className = "log-flag " + (entry.allowed ? "ok" : "no");
            flag.textContent = entry.allowed ? copyAt("labels.audit.pass") : copyAt("labels.audit.deny");
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
            box.innerHTML = '<p class="empty"></p>';
            box.firstChild.textContent = copyAt("safety.audit.empty");
            return;
        }
        state.audit.forEach(function (entry) {
            var row = document.createElement("div");
            row.className = "audit-row";

            var time = document.createElement("span");
            time.textContent = entry.time;

            var flag = document.createElement("span");
            flag.className = entry.allowed ? "ok" : "no";
            flag.textContent = entry.allowed ? copyAt("labels.audit.pass") : copyAt("labels.audit.deny");

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

    function tableLabel(name) {
        return (COPY.labels.tables || {})[name] || name;
    }

    function plainJoin(key) {
        var m = String(key).match(/^(\w+)\.(\w+) = (\w+)\.(\w+)$/);
        if (!m) return String(key);
        return fmt(copyAt("messages.joinTemplate"), {
            a: tableLabel(m[1]),
            colA: m[2],
            b: tableLabel(m[3]),
            colB: m[4]
        });
    }

    function applyRewrites(text, rewrites) {
        return (rewrites || []).reduce(function (out, pair) {
            return out.split(pair.from).join(pair.to);
        }, String(text));
    }

    function plainScenarioText(text) {
        return applyRewrites(text, COPY.schemaRewrites && COPY.schemaRewrites.scenario);
    }

    function plainPitfall(text) {
        return applyRewrites(plainScenarioText(text),
            COPY.schemaRewrites && COPY.schemaRewrites.pitfalls);
    }
}());
