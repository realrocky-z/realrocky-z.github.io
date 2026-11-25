# Obsidian → 个人主页 Blog 发布流程设计

状态：设计稿。现有网站下线调整已实现；下述 Obsidian 导出器尚未实现，不应把现有 publish.py blog 的直接复制视为支持双链。

## 1. 选择与当前基础

保留现有静态站、Python Markdown 构建器与 GitHub Pages，不另建 Quartz 站。这样知识笔记与现有 Blog 共用地址、样式和部署。

目前 scripts/publish.py blog 只复制一篇 Markdown，不处理 Obsidian 双链和附件；scripts/build_blog.py 处理普通 Markdown、目录与数学公式；main 分支推送触发 .github/workflows/deploy.yml。

用户已指定“本科课程学习文档汇总”为源库，按现有 A专业课、辅修专业、C学习拓展等课程目录就地标记发布，无需迁移到 D:/obsidian。由本机配置指定实际 vault，不把个人绝对路径写进公开仓库。原库始终是唯一编辑源，不要求移动笔记，也不将整库提交到网站仓库。

## 2. 日常操作

1. 在 Obsidian 写学习笔记，照常使用 [[知识点]]、[[知识点#标题|显示文字]]。
2. 选择公开的文章和关联知识点，补全下列 Properties。只想支持文章知识链接、无需出现在 Blog 列表的知识点使用 publish_role: reference。
3. 执行拟新增命令 `python scripts/export_obsidian.py --config publish.local.yml --check`，查看待新增、更新、撤回、附件和断链清单。此命令不改站点。
4. 执行 `python scripts/export_obsidian.py --config publish.local.yml`，仅同步显式公开的笔记；构建文章并本地预览，检查公式、图片、锚点及反向链接。
5. 查看 Git diff，只提交本次导出与必要配置。推送 main 后由现有 GitHub Actions 构建部署。第一阶段采用手动触发，不要每次保存笔记就公开。
6. 等待 Pages 部署成功，检查真实站点的文章 URL、知识跳转、移动端和图片，再确认上线完成。构建成功不等于部署成功。

上述 export_obsidian.py、publish.local.yml 是拟新增接口，目前不可直接运行。现有构建命令仍为 `python scripts/build_blog.py`。

## 3. 笔记发布协议

```yaml
---
title: 条件期望为什么是最优预测
date: 2026-11-25
summary: 从平方损失解释条件期望，并链接到相关知识点。
publish: true
publish_role: post
slug: conditional-expectation
publish_id: note-conditional-expectation-001
draft: false
tags: [概率论, 统计学习]
---
```

- 只有 `publish: true` 且 `draft: false` 同时满足才允许导出；缺少字段默认私有。post 必须有标题、有效日期、摘要、唯一 slug、唯一 publish_id。
- post 是 Blog 正文；reference 是公开的关联知识卡片，有自己的页面但不加入 Blog 主列表。reference 仍需明确 publish，不自动公开被引用笔记。
- slug 首次发布后保持不变，页面地址为 /blog/<slug>.html。重命名或移动 Obsidian 文件不会改变 URL；publish_id 用于追踪源文件。变更 slug 时必须提供旧地址重定向记录，防止外链失效。
- date 是首次发布日期，updated 记录修订日期。不以文件修改时间重排所有历史文章。
- 网站中的 blog/_obsidian/ 是受导出器管理的产物目录；手写 blog/*.md 保留现有流程。导出器只更新 manifest 中属于自己的文件，不覆盖手写稿。

## 4. 保留知识链接的规则

使用 Markdown 语法树和 Obsidian 专用 token 解析，不对整篇文本做粗暴正则替换，避免改坏代码块、行内代码、数学公式和外部 URL。

| Obsidian 原文 | 网页行为 |
|---|---|
| [[条件期望]] | 解析到目标笔记的稳定 slug，生成 /blog/conditional-expectation.html |
| [[条件期望\|最优预测]] | 保留显示文字“最优预测”，地址同上 |
| [[条件期望#投影解释]] | 目标网页加标题锚点；锚点由与渲染器共享的映射产生 |
| [[#当前标题]] | 跳转本页标题；重复标题按渲染器实际生成 ID 区分 |
| [[条件期望#^projection]] | 将 Obsidian 块 ID 转为网页 id="block-projection"，链接到该位置 |
| ![[示意图.png]] | 将被公开笔记引用的附件复制到 assets/notes/<内容哈希>.<扩展名>，保留图片显示与宽度 |
| 普通 Markdown 的相对链接/图片 | 按源笔记所在目录解析，再重写为网站地址；保留网络链接 |
| [[尚未公开的笔记]] | 阻止发布，指出源文件与行号；由作者选择公开目标、去掉链接或改写文字 |
| ![[另一篇笔记]] / 标题或块嵌入 | 第一版明确报“不支持转录”，阻止发布；后续加入受控展开、循环检测后再支持 |

目标解析优先精确相对路径与库内完整路径，再匹配唯一文件名和 aliases；重名或 alias 冲突一律报错，不能猜测。中文、空格、# 和 URL 编码由统一 URL 层处理。别名、标题和块 ID 都需要唯一性检查。附件与符号链接解析后的真实路径必须处于允许的 vault/附件目录内，禁止通过 ../ 读取库外文件。

为每个公开页面生成“被哪些笔记引用”区域：导出时保存链接图，构建时生成反向链接。reference 页面可通过“相关知识”访问，不挤占 Blog 文章列表。知识网络第一阶段依赖链接和反向链接即可，无需先做交互式图谱。

## 5. 管道分层与落地改动

```text
Obsidian 原库（私有）
  → 本地扫描 Properties，生成路径/别名/publish_id 索引
  → 校验公开集合、重名、链接、锚点、附件
  → 增量导出 Markdown + 附件 + manifest + 链接图
  → build_blog.py 渲染文章/知识页、目录和反向链接
  → 本地预览 + Git diff
  → main 推送
  → GitHub Actions 校验、构建公共包、部署 Pages
  → 线上冒烟检查
```

拟新增：

- scripts/export_obsidian.py：读取本地配置、两遍解析、check 模式、暂存导出、确认全部校验成功后再更新产物，避免半次导出。
- publish.local.example.yml：示例配置；真实 publish.local.yml 加入 .gitignore。
- blog/_obsidian/manifest.json：publish_id → slug、源相对标识、内容哈希、产物与附件所有权；不含私有路径或未公开标题。
- scripts/check_blog_links.py：在 CI 校验所有内部页面、标题、块锚点及附件存在；私有源库不进入 CI。
- scripts/build_blog.py：递归读取导出目录、严格验证稳定 slug、重复 slug 报错（不再静默加 -2）、区分 post/reference、生成反向链接及公开页面清单。
- scripts/build_public_site.py：按公开页面清单收录 post 和 reference。目前使用 Blog 列表作为白名单；加入 reference 时必须同步扩展清单，避免知识页被误删。
- .github/workflows/deploy.yml：渲染后运行链接校验，再上传公共产物。CI 不连接私人 Obsidian 库。

导出器遇到取消 publish 的笔记，应报告撤回并仅删除自己 manifest 中的产物；剩余公开文章仍引用它时阻止发布。附件按引用计数清理。撤回必须同时从列表、链接图和公共部署包中移除，不能只隐藏入口。Git 历史不因撤回自动消失，因此导出前审阅是必要步骤。

## 6. 上线门槛

用两篇文章和两张知识卡片作为试点，覆盖：中文文件名、带别名双链、跨目录链接、标题/重复标题、块链接、图片宽度、相对 Markdown 图片、公式、代码中出现 [[...]] 不被替换。另验证私有目标、重名目标、缺失附件、无效 slug、嵌入笔记都会明确失败。

验收还包括：源文件移动但 URL 不变；二次导出没有无意义 diff；撤回不会残留旧网页；导出失败不损坏上一版；公开产物没有 .obsidian、私有笔记和本机路径。最后在真实 Pages 域名验证跳转与一次回滚。

建议分两轮实施：第一轮完成显式发布、双链/标题/块链接、附件、断链检查、手动部署；第二轮加入 Obsidian 命令入口、一键预览以及受控笔记嵌入。不要先引入实时双向同步，避免网页编辑与 Obsidian 源稿发生冲突。
