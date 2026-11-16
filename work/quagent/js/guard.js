/* sqlkit.guard 的浏览器移植：单条语句、只读关键字、注释与字符串边界。 */
(function (root, factory) {
    if (typeof module === "object" && module.exports) {
        module.exports = factory();
    } else {
        root.QuagentGuard = factory();
    }
}(typeof self !== "undefined" ? self : this, function () {
    "use strict";

    var FORBIDDEN = /\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|DETACH|PRAGMA|VACUUM|REINDEX|REPLACE|TRUNCATE|GRANT|REVOKE)\b/i;
    var ALLOWED_FIRST = /^(SELECT|WITH|EXPLAIN)$/i;

    function copyGroup() {
        var host = typeof self !== "undefined" ? self : (typeof globalThis !== "undefined" ? globalThis : null);
        return host && host.QUAGENT_COPY ? host.QUAGENT_COPY.guard || null : null;
    }

    function text(key, fallback, values) {
        var group = copyGroup();
        var template = group && typeof group[key] === "string" ? group[key] : fallback;
        return template.replace(/\{(\w+)\}/g, function (_, name) {
            return values && values[name] != null ? values[name] : "";
        });
    }

    function scan(sql) {
        var clean = [];
        var masked = [];
        var quote = null;
        var i = 0;
        var n = sql.length;

        function mask(ch) {
            masked.push(/\s/.test(ch) ? ch : " ");
        }

        while (i < n) {
            var ch = sql[i];
            var next = i + 1 < n ? sql[i + 1] : "";

            if (quote) {
                clean.push(ch);
                mask(ch);
                if (quote === "[") {
                    if (ch === "]") { quote = null; }
                    i += 1;
                    continue;
                }
                if (ch === quote) {
                    if (next === quote) {
                        clean.push(next);
                        mask(next);
                        i += 2;
                        continue;
                    }
                    quote = null;
                }
                i += 1;
                continue;
            }

            if (ch === "-" && next === "-") {
                while (i < n && sql[i] !== "\n") { i += 1; }
                clean.push(" ");
                masked.push(" ");
                continue;
            }

            if (ch === "#") {
                while (i < n && sql[i] !== "\n") { i += 1; }
                clean.push(" ");
                masked.push(" ");
                continue;
            }

            if (ch === "/" && next === "*") {
                var end = sql.indexOf("*/", i + 2);
                if (end === -1) {
                    return { error: text("unclosedComment", "注释没有写完。") };
                }
                clean.push(" ");
                masked.push(" ");
                i = end + 2;
                continue;
            }

            if (ch === "'" || ch === '"' || ch === "`" || ch === "[") {
                quote = ch;
                clean.push(ch);
                mask(ch);
                i += 1;
                continue;
            }

            clean.push(ch);
            masked.push(ch);
            i += 1;
        }

        if (quote) {
            return {
                error: quote === "["
                    ? text("unclosedBracket", "方括号没有配对。")
                    : text("unclosedQuote", "引号没有配对。")
            };
        }

        return {
            clean: clean.join(""),
            masked: masked.join("")
        };
    }

    function splitStatements(clean, masked) {
        var parts = [];
        var start = 0;
        for (var i = 0; i < masked.length; i += 1) {
            if (masked[i] === ";") {
                parts.push(clean.slice(start, i));
                start = i + 1;
            }
        }
        parts.push(clean.slice(start));
        return parts.map(function (p) { return p.trim(); }).filter(Boolean);
    }

    function guard(sql) {
        var raw = String(sql == null ? "" : sql);
        if (raw.indexOf("\u0000") !== -1) {
            return { ok: false, sql: "", error: text("nullChar", "语句里含有无法识别的空字符。") };
        }

        var scanned = scan(raw);
        if (scanned.error) {
            return { ok: false, sql: "", error: scanned.error };
        }

        var parts = splitStatements(scanned.clean, scanned.masked);
        if (parts.length !== 1) {
            return {
                ok: false,
                sql: "",
                error: text("multiStatement", "一次只能提交一条语句，这里实际有 {n} 条。", { n: parts.length })
            };
        }

        var one = parts[0];
        var maskedOne = scanned.masked;
        var first = maskedOne.match(/^\s*([A-Za-z]+)/);
        if (!first || !ALLOWED_FIRST.test(first[1])) {
            return {
                ok: false,
                sql: one,
                error: text("notQuery", "只能执行查询语句，查询要以 SELECT、WITH 或 EXPLAIN 开头。")
            };
        }

        var bad = maskedOne.match(FORBIDDEN);
        if (bad) {
            return {
                ok: false,
                sql: one,
                error: text(
                    "writeKeyword",
                    "这条语句会修改或删除数据（{op}），已经拒绝。",
                    { op: bad[1].toUpperCase() }
                )
            };
        }

        return { ok: true, sql: one, error: "" };
    }

    return { guard: guard };
}));
