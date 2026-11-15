/* sql.js 执行单元：只读编译、限行执行、错误回传。 */
importScripts("../vendor/sql-wasm.js");

var SQL = null;
var db = null;

var ready = initSqlJs({
    locateFile: function (file) {
        return new URL("../vendor/" + file, self.location.href).href;
    }
});

function reply(id, ok, payload) {
    self.postMessage({ id: id, ok: ok, payload: payload });
}

self.onmessage = async function (event) {
    var msg = event.data;
    var id = msg.id;
    try {
        if (msg.cmd === "init") {
            SQL = await ready;
            if (db) {
                db.close();
                db = null;
            }
            db = new SQL.Database(new Uint8Array(msg.buffer));
            db.run("PRAGMA query_only = ON");
            reply(id, true, { tables: 6 });
            return;
        }

        if (!db) {
            reply(id, false, { error: "示例数据还没有加载。" });
            return;
        }

        if (msg.cmd === "validate") {
            var sqlText = String(msg.sql || "");
            var prefix = /^\s*EXPLAIN\b/i.test(sqlText) ? "" : "EXPLAIN QUERY PLAN ";
            var plan = db.exec(prefix + sqlText);
            reply(id, true, {
                plan: plan.length ? plan[0].values : [],
                columns: plan.length ? plan[0].columns : []
            });
            return;
        }

        if (msg.cmd === "run") {
            var limit = Math.max(1, Math.min(Number(msg.limit) || 200, 1000));
            var started = performance.now();
            var stmt = db.prepare(msg.sql);
            var columns = stmt.getColumnNames();
            var rows = [];
            while (rows.length <= limit && stmt.step()) {
                rows.push(stmt.get());
            }
            stmt.free();
            var truncated = rows.length > limit;
            if (truncated) {
                rows = rows.slice(0, limit);
            }
            reply(id, true, {
                columns: columns,
                rows: rows,
                row_count: rows.length,
                truncated: truncated,
                elapsed_ms: Math.round((performance.now() - started) * 100) / 100
            });
            return;
        }

        reply(id, false, { error: "未知命令：" + msg.cmd });
    } catch (err) {
        reply(id, false, { error: err && err.message ? err.message : String(err) });
    }
};
