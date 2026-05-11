# Worklog

## 2026-05-11 — 第一期：反直觉简报管线搭建

### 目标

构建一个自动管线：搜索 AI 资讯 → 提炼 3 个反直觉点 → 事实核查 → 输出 HTML。

### 尝试记录

**1. 提示词实验（直接发生在对话中）**

- 尝试 1：让 AI 搜 Karpathy 资讯"总结给我" → 输出平平无奇
- 尝试 2：改为"其中有什么反直觉的点，给我 3 个" → 输出质量明显提升
- 结论：prompt 中给筛选标准（反直觉）比给收集指令（总结）效果好

**2. pi 提示词结构设计**

- 设计了 5 阶段管线：搜集素材 → 初稿 → codex 审查 → 修正 → 发送
- 关键决策：dryrun 时从提示词中物理删除阶段 5（发邮件），不留条件判断
- 关键决策：发送逻辑归 ps1 管理，pi 只负责内容产出

**3. HTML 报纸风格（3 次迭代）**

- v1：纯文字报纸风格 → 舰长要求加入视觉元素
- v2：SVG 箭头关系图谱 → 舰长不满意，回退到 v1（暴露无 git 的问题）
- v3：回到纯文字版 + git init（第一次 commit）
- 结论：无 git 时改版不可回退，需先初始化版本控制

**4. 图片生成方向探索**

- 方向 1（箭头关系图）：codex image_gen 生成，质量可用但信息密度低
- 方向 2（比喻插画）：codex image_gen 生成，舰长确认方向对
- 后续结论：codex 自主发挥的创意不稳定，需人类提供具体构图

**5. 图片生成提示词策略迭代**

- 尝试 1：给 1-2 句摘要 + 风格指令 → codex 发挥过度，画面莫名其妙
- 尝试 2：调研 + 完整原文 + 开放创意 → 仍然不稳定
- 尝试 3：调研 + 完整原文 + 人类提供具体创意（画面元素+文字+关系） → 效果可控
- 结论：生图环节目前必须人类提供具体创意，无法全自动

**6. 三张图的生成**

- cartoon-01.png：火车+观察员（时间线翻转）— 第一次失败（自主创意莫名其妙），第二次人类给具体创意后成功
- cartoon-02.png：畸形大脑幽灵（jagged intelligence）— 创意由人类提供（"刷爆所有 Benchmark" vs "建议把汽车装进口袋"），一次成功
- cartoon-03.png：脚手架+高塔（互补/盲区）— 前5个idea被否定（全落回"高失败率+偶然发现+被质疑"套路），改方向后人类选定idea2（互补脚手架），生成成功但画面偏复杂

**7. 图片压缩**

- 测试了4种方案：原PNG / 隔行扫描 / 256色调色板 / JPEG q85
- 结论：选择 JPEG q85，三张图 7.8MB → 1.2MB（14-18%），黑色线条无明显伪影
- EXIF 信息不存在，无需清理

**8. HTML 排版调整**

- 图在文前：figure 移到 article 之前（先看漫画再看文字）
- 删除底部方向1/方向2 gallery（实验废稿）
- figcaption 去重：三张图各自独特的描述文字
- colophon 修改："每日自动生成" → "人类提供漫画创意"（诚实反映当前状态）
- 移动端适配：body padding=0，datebar flex-direction:column 竖向换行

**9. 部署**

- 研究现有 io99.xyz 子域项目的 wrangler.toml 配置（fuyou/bazi/gold-price-worker）
- 创建 wrangler.toml + public/ 目录，纯静态资源（无 worker）
- 部署到 news.io99.xyz（Cloudflare Workers）

**10. Git 与 GitHub**

- gh repo push 遇到分支名问题：本地 master → 远程默认 main，强推后远程仍显示旧内容
- 修复：git branch -m master main，再 force push 到 main
- news 仓库名已存在但为不同项目，force push 覆盖
- 添加 .gitignore（排除 .wrangler/）

### 当前状态

- 管线可跑通：pi → codex 审查 → pi 修正 → 人类提供漫画创意 → codex 生图 → 嵌入 HTML → wrangler 部署
- 文字部分可自动，图片部分需人类提供具体创意
- 第一期已部署到 news.io99.xyz
- 移动端已适配

### 未解决问题

- 图片生成无法自动化（需人类给具体构图创意）
- 从 final.md 到 HTML（含漫画+排版）的自动化尚未实现
- gh repo push 时 master/main 分支名不匹配导致初次推送失败
- live-server 在 PowerShell 后台 job 中不稳定，改为直接打开文件
