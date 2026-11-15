import test from "node:test";
import assert from "node:assert/strict";
import guardModule from "../js/guard.js";

const guard = guardModule.guard;

test("允许单条 SELECT 与大小写", () => {
    assert.equal(guard("SELECT 1").ok, true);
    assert.equal(guard("  select 1  ").ok, true);
    assert.equal(guard("WITH x AS (SELECT 1) SELECT * FROM x").ok, true);
    assert.equal(guard("EXPLAIN SELECT 1").ok, true);
});

test("允许结尾分号", () => {
    const out = guard("SELECT 1;");
    assert.equal(out.ok, true);
    assert.equal(out.sql, "SELECT 1");
});

test("拒绝多语句", () => {
    const out = guard("SELECT 1; SELECT 2");
    assert.equal(out.ok, false);
    assert.match(out.error, /2 条/);
});

test("拒绝写操作与危险关键字", () => {
    assert.equal(guard("DROP TABLE orders").ok, false);
    assert.equal(guard("delete from orders").ok, false);
    assert.equal(guard("WITH x AS (SELECT 1) DELETE FROM orders").ok, false);
    assert.equal(guard("EXPLAIN DELETE FROM orders").ok, false);
    assert.equal(guard("PRAGMA table_info(orders)").ok, false);
});

test("拒绝非 SELECT/WITH/EXPLAIN 开头", () => {
    const out = guard("VALUES (1)");
    assert.equal(out.ok, false);
    assert.match(out.error, /SELECT、WITH 或 EXPLAIN/);
});

test("引号内的分号与注释标记不误判", () => {
    assert.equal(guard("SELECT ';' AS s").ok, true);
    const out = guard("SELECT '--x' AS s");
    assert.equal(out.ok, true);
    assert.match(out.sql, /--x/);
    assert.equal(guard("SELECT '/*x*/' AS s").ok, true);
});

test("引号内的关键字不误杀", () => {
    assert.equal(guard("SELECT 'delete' AS s").ok, true);
    assert.equal(guard('SELECT "drop" FROM orders').ok, true);
    assert.equal(guard("SELECT [delete] FROM orders").ok, true);
    assert.equal(guard("SELECT 1 FROM orders WHERE status = 'DROP'").ok, true);
});

test("注释被剥离且注释内分号不切句", () => {
    const out = guard("/* c */ SELECT 1 /* ; */");
    assert.equal(out.ok, true);
    assert.equal(out.sql, "SELECT 1");
    assert.equal(guard("SELECT 1 -- hi").ok, true);
    assert.equal(guard("SELECT 1 # hi").ok, true);
    assert.equal(guard("SELECT 1 /* ; */ ; SELECT 2").ok, false);
});

test("未闭合引号与注释被拒", () => {
    assert.equal(guard("SELECT 'x").ok, false);
    assert.equal(guard("SELECT \"x").ok, false);
    assert.equal(guard("SELECT 1 /*").ok, false);
});

test("空语句与空字符被拒", () => {
    assert.equal(guard("").ok, false);
    assert.equal(guard("   ").ok, false);
    assert.equal(guard("SELECT 1" + String.fromCharCode(0)).ok, false);
});

test("双引号转义按 SQLite 规则处理", () => {
    assert.equal(guard("SELECT 'it''s fine' AS s").ok, true);
    assert.equal(guard('SELECT "a""b" FROM orders').ok, true);
});
