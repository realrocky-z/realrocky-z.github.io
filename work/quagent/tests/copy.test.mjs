import test from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import vm from "node:vm";
import { fileURLToPath } from "node:url";

const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), "..");
const read = (rel) => fs.readFileSync(path.join(root, rel), "utf8");

function loadCopy() {
    const sandbox = { window: {} };
    sandbox.window.window = sandbox.window;
    vm.runInNewContext(read("js/copy.js"), sandbox, { filename: "copy.js" });
    return sandbox.window.QUAGENT_COPY;
}

function at(node, key) {
    return key.split(".").reduce((value, part) => {
        return value == null ? undefined : value[part];
    }, node);
}

const copy = loadCopy();

test("index.html 引用的文案键全部存在且非空", () => {
    const html = read("index.html");
    const keys = [...html.matchAll(/data-copy(?:-html|-placeholder|-aria)?="([^"]+)"/g)]
        .map((m) => m[1]);
    assert.ok(keys.length > 0, "index.html 没有任何 data-copy 键");
    for (const key of keys) {
        const value = at(copy, key);
        assert.equal(typeof value, "string", "缺少文案键：" + key);
        assert.ok(value.trim().length > 0, "文案键为空：" + key);
    }
});

test("24 道题都有题干，数据文件只保留机器字段", () => {
    const questions = JSON.parse(read("data/questions.json"));
    assert.equal(questions.length, 24);
    assert.equal(Object.keys(copy.questions).length, 24);
    for (const q of questions) {
        assert.equal("question" in q, false, q.id + " 仍含 question 字段");
        const text = copy.questions[q.id];
        assert.equal(typeof text, "string", "缺少题干：" + q.id);
        assert.ok(text.trim().length > 0, "题干为空：" + q.id);
    }
});

test("指标说明已并入文案文件", () => {
    const html = copy.scenario.metricsHtml;
    assert.equal(typeof html, "string");
    assert.match(html, /指标口径/);
    assert.ok(html.length > 500, "指标说明过短，可能没有迁移完整");
    assert.equal(fs.existsSync(path.join(root, "data/metrics.html")), false);
});

test("hero 步骤、安全层、工具与示例都是非空数组", () => {
    assert.equal(copy.hero.steps.length, 4);
    for (const step of copy.hero.steps) {
        assert.ok(step.trim().length > 0);
    }

    assert.equal(copy.safety.layers.length, 4);
    for (const layer of copy.safety.layers) {
        assert.ok(layer.no && layer.title && layer.bodyHtml && layer.evidenceHtml);
    }

    assert.equal(copy.tools.length, 6);
    for (const tool of copy.tools) {
        assert.ok(tool.name && tool.desc);
    }

    assert.equal(copy.examples.length, 6);
    for (const example of copy.examples) {
        assert.ok(example.title && example.sql && example.note);
    }
});

test("guard、compare、worker 的提示语齐全", () => {
    const groups = {
        guard: 7,
        compare: 12,
        worker: 2
    };
    for (const [name, count] of Object.entries(groups)) {
        const group = copy[name];
        assert.equal(typeof group, "object", "缺少文案组：" + name);
        const values = Object.values(group);
        assert.equal(values.length, count, name + " 的提示条数不对");
        for (const value of values) {
            assert.equal(typeof value, "string");
            assert.ok(value.trim().length > 0, name + " 里有空提示");
        }
    }
});
