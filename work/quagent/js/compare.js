/* check_answers.py 的浏览器移植：列数、行数、多重集合、数值允许 0.01 的误差。 */
(function (root, factory) {
    if (typeof module === "object" && module.exports) {
        module.exports = factory();
    } else {
        root.QuagentCompare = factory();
    }
}(typeof self !== "undefined" ? self : this, function () {
    "use strict";

    var TOL = 0.01;

    function copyGroup() {
        var host = typeof self !== "undefined" ? self : (typeof globalThis !== "undefined" ? globalThis : null);
        return host && host.QUAGENT_COPY ? host.QUAGENT_COPY.compare || null : null;
    }

    function text(key, fallback, values) {
        var group = copyGroup();
        var template = group && typeof group[key] === "string" ? group[key] : fallback;
        return template.replace(/\{(\w+)\}/g, function (_, name) {
            return values && values[name] != null ? values[name] : "";
        });
    }

    function normalize(value) {
        if (value === null || value === undefined) {
            return { type: "null", value: "" };
        }
        if (typeof value === "boolean") {
            return { type: "num", value: value ? 1 : 0 };
        }
        if (typeof value === "number") {
            return { type: "num", value: value };
        }
        return { type: "str", value: String(value).trim() };
    }

    function sameValue(a, b) {
        var x = normalize(a);
        var y = normalize(b);
        if (x.type === "null" || y.type === "null") {
            return x.type === "null" && y.type === "null";
        }
        if (x.type === "num" && y.type === "num") {
            return Math.abs(x.value - y.value) <= TOL;
        }
        return String(x.value) === String(y.value);
    }

    function sameRow(a, b) {
        if (!a || !b || a.length !== b.length) {
            return false;
        }
        for (var i = 0; i < a.length; i += 1) {
            if (!sameValue(a[i], b[i])) {
                return false;
            }
        }
        return true;
    }

    function sameRows(aRows, bRows, ordered) {
        if (aRows.length !== bRows.length) {
            return false;
        }
        if (ordered) {
            for (var i = 0; i < aRows.length; i += 1) {
                if (!sameRow(aRows[i], bRows[i])) {
                    return false;
                }
            }
            return true;
        }
        var remaining = bRows.slice();
        for (var r = 0; r < aRows.length; r += 1) {
            var found = -1;
            for (var j = 0; j < remaining.length; j += 1) {
                if (sameRow(aRows[r], remaining[j])) {
                    found = j;
                    break;
                }
            }
            if (found === -1) {
                return false;
            }
            remaining.splice(found, 1);
        }
        return true;
    }

    function grade(result, golden) {
        var columns = result.columns || [];
        var rows = result.rows || [];
        var expect = golden.expect_columns || [];
        var goldRows = golden.golden_rows || [];

        if (columns.length !== expect.length) {
            return {
                pass: false,
                reason: text("columnMismatch", "列数对不上"),
                detail: text(
                    "columnDetail",
                    "你返回了 {got} 列，这道题需要 {want} 列。",
                    { got: columns.length, want: expect.length }
                )
            };
        }
        if (result.truncated) {
            return {
                pass: false,
                reason: text("tooManyReason", "结果太多了"),
                detail: text("tooManyDetail", "查询返回的行数超过页面能比较的数量，没法完整比对。")
            };
        }
        if (rows.length !== goldRows.length) {
            return {
                pass: false,
                reason: text("rowMismatch", "行数对不上"),
                detail: text(
                    "rowDetail",
                    "你返回了 {got} 行，这道题需要 {want} 行。",
                    { got: rows.length, want: goldRows.length }
                )
            };
        }
        if (!sameRows(rows, goldRows, Boolean(golden.ordered))) {
            return {
                pass: false,
                reason: text("resultMismatch", "结果对不上"),
                detail: firstDiff(rows, goldRows, Boolean(golden.ordered))
            };
        }
        return {
            pass: true,
            reason: text("passReason", "通过"),
            detail: text("passDetail", "列数和每一行的结果都对上了。")
        };
    }

    function firstDiff(rows, goldRows, ordered) {
        if (ordered) {
            for (var i = 0; i < rows.length; i += 1) {
                if (!sameRow(rows[i], goldRows[i])) {
                    return text(
                        "rowDiff",
                        "第 {n} 行不一样。你的是 {got}，标准答案是 {want}。",
                        { n: i + 1, got: show(rows[i]), want: show(goldRows[i]) }
                    );
                }
            }
        }
        var remaining = goldRows.slice();
        for (var r = 0; r < rows.length; r += 1) {
            var found = -1;
            for (var j = 0; j < remaining.length; j += 1) {
                if (sameRow(rows[r], remaining[j])) {
                    found = j;
                    break;
                }
            }
            if (found === -1) {
                return text(
                    "rowMissing",
                    "你返回的 {row} 这一行不在标准答案里。",
                    { row: show(rows[r]) }
                );
            }
            remaining.splice(found, 1);
        }
        return text("resultDetail", "结果和标准答案对不上。");
    }

    function show(row) {
        return "[" + row.map(function (v) {
            return v === null ? "NULL" : String(v);
        }).join(", ") + "]";
    }

    return {
        grade: grade,
        sameValue: sameValue,
        sameRow: sameRow,
        sameRows: sameRows
    };
}));
