import test from "node:test";
import assert from "node:assert/strict";
import compareModule from "../js/compare.js";

const { grade, sameValue, sameRow, sameRows } = compareModule;

const golden = {
    expect_columns: ["city", "orders", "net_gmv"],
    ordered: false,
    golden_rows: [
        ["Singapore", 10, 123.45],
        ["Bangkok", 8, 100.0]
    ]
};

test("无序比对允许行顺序不同", () => {
    const out = grade({
        columns: ["a", "b", "c"],
        rows: [["Bangkok", 8, 100], ["Singapore", 10, 123.45]]
    }, golden);
    assert.equal(out.pass, true);
});

test("有序比对要求行顺序一致", () => {
    const g = { ...golden, ordered: true };
    const out = grade({
        columns: ["a", "b", "c"],
        rows: [["Bangkok", 8, 100], ["Singapore", 10, 123.45]]
    }, g);
    assert.equal(out.pass, false);
    assert.match(out.detail, /第 1 行/);
});

test("数值相差 0.01 以内算一致", () => {
    assert.equal(sameValue(123.45, 123.46), true);
    assert.equal(sameValue(123.45, 123.47), false);
});

test("NULL 只匹配 NULL", () => {
    assert.equal(sameValue(null, null), true);
    assert.equal(sameValue(null, 0), false);
    assert.equal(sameValue(null, ""), false);
});

test("字符串去首尾空格后比较", () => {
    assert.equal(sameValue(" foo ", "foo"), true);
});

test("列数不一致直接失败", () => {
    const out = grade({
        columns: ["a", "b"],
        rows: [["Singapore", 10]]
    }, golden);
    assert.equal(out.pass, false);
    assert.match(out.detail, /你返回了 2 列，这道题需要 3 列/);
});

test("行数不一致直接失败", () => {
    const out = grade({
        columns: ["a", "b", "c"],
        rows: [["Singapore", 10, 123.45]]
    }, golden);
    assert.equal(out.pass, false);
    assert.match(out.detail, /你返回了 1 行，这道题需要 2 行/);
});

test("多重集合按重复次数匹配", () => {
    assert.equal(sameRows([[1], [1]], [[1], [2]], false), false);
    assert.equal(sameRows([[1], [1]], [[1], [1]], false), true);
});

test("行内值逐个比较", () => {
    assert.equal(sameRow([1, "a", null], [1.0, " a ", null]), true);
    assert.equal(sameRow([1, "a"], [1, "b"]), false);
});
