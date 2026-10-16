# AI Memory Sync

> Mnemopi ↔ Obsidian Vault 双向自动同步扩展 — AI 自主知识领域聚类 + MOC 索引 + 向量语义检索

[Oh My Pi](https://github.com/earendil-works/pi-coding-agent) 编程助手的记忆扩展插件。将 AI 对话中产生的长期记忆（通过 Mnemopi 存储）自动同步到 Obsidian vault，并按知识领域自动聚类，生成 MOC (Map of Content) 索引笔记，让 Obsidian 图谱视图呈现清晰的聚类结构。

## 功能

### 双向同步
- **自动导出**: `retain` / `learn` 写入记忆后，自动增量导出为结构化 Markdown 到 Obsidian vault
- **自动导入**: `inbox/` 目录下的笔记在每回合结束时自动导入回 Mnemopi
- **Git 同步**: 导出后自动 `git commit + push`（3 次重试，push 失败不中断）

### 知识领域化分类（AI 自主聚类）
- **无硬编码领域**: 不预设领域分类，AI 在记忆量达到阈值后用嵌入向量做 K-means++ 聚类自主发现领域
- **MOC 索引笔记**: 每个领域生成 `_moc/<domain>.md` hub 笔记，列出该域所有记忆的 wikilinks
- **Obsidian 聚类图**: 每条记忆通过 `[[_moc/<domain>]]` wikilink 连接到 MOC hub，图谱视图中同域记忆形成聚类团
- **全局索引**: `_moc/_index.json` 汇总所有领域信息

### 语义检索
- **`memory_recall` 工具**: 模型可主动调用向量化语义检索，返回 top-K 相关记忆
- **主动检索注入**: 回合开始时自动检索相关记忆并注入上下文（proactive retrieval）

### 结构化记忆
- **YAML frontmatter**: 每条记忆带 `id`/`source`/`importance`/`domain`/`keywords`/`tags`
- **语义分块**: 长记忆按段落+句号分块（≤500 字符/块），每块独立检索
- **增量导出**: 用 `rowid` 游标避免相同时刻漏导

## 安装

```bash
# 1. 克隆仓库
git clone https://github.com/NekoMirra/ai-memory-sync.git

# 2. 复制到 omp 扩展目录
cp -r ai-memory-sync ~/.omp/agent/extensions/ai-memory-sync

# 3. 安装依赖与创建配置文件
cd ~/.omp/agent/extensions/ai-memory-sync
npm install
cp config.example.json config.json
# 编辑 config.json 填入你的路径和 API key

omp 启动时自动发现并加载扩展。

## 配置

编辑 `config.json`:

```json
{
  "vaultDir": "/path/to/your/obsidian/vault",
  "mnemopiDb": "/path/to/mnemopi.db",
  "debounceMs": 3000,
  "gitAutoCommit": true,
  "gitAutoPush": true,
  "embeddingModel": "@cf/baai/bge-m3",
  "embeddingApiUrl": "https://api.cloudflare.com/client/v4/accounts/<ACCOUNT_ID>/ai/v1",
  "embeddingApiKey": "<YOUR_API_KEY>",
  "proactiveRecall": true,
  "proactiveTopK": 5,
  "domainClusterThreshold": 30
}
```

| 字段 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `vaultDir` | string | — | Obsidian vault 根目录 |
| `mnemopiDb` | string | — | Mnemopi SQLite 数据库路径 |
| `debounceMs` | number | `3000` | 防抖延迟（毫秒） |
| `gitAutoCommit` | boolean | `true` | 导出后自动 git commit |
| `gitAutoPush` | boolean | `true` | commit 后自动 git push |
| `embeddingModel` | string | `@cf/baai/bge-m3` | 嵌入模型名称 |
| `embeddingApiUrl` | string | — | 嵌入 API 地址 |
| `embeddingApiKey` | string | — | 嵌入 API 密钥 |
| `proactiveRecall` | boolean | `true` | 回合开始时主动检索注入 |
| `proactiveTopK` | number | `5` | 主动检索返回数量 |
| `domainClusterThreshold` | number | `30` | 触发聚类的最小未分类记忆数 |

## Vault 目录结构

```
your-vault/
├── brain/
│   └── memories/
│       ├── embedded-esp32/     ← AI 聚类发现的领域目录
│       │   ├── 2026-08-27-..._a1b2c3d4.md
│       │   └── ...
│       ├── dify-workflow/
│       ├── reverse-engineering/
│       └── ...
├── _moc/
│   ├── _index.json             ← 全局领域索引
│   ├── embedded-esp32.md       ← MOC hub 笔记（wikilinks 聚类）
│   ├── dify-workflow.md
│   └── ...
├── inbox/                      ← 放入 .md 文件会自动导入 Mnemopi
└── brain/imported/             ← 导入后的文件移到这里
```

## 导出的 Markdown 格式

```yaml
---
id: "2fe11c1952ba7e21"
source: "coding-agent-learn"
importance: 0.8
timestamp: "2026-08-26 09:58:45"
scope: "bank"
domain: "embedded-esp32"
keywords: ["esp32", "nuttx", "openvela", "mipi"]
tags: ["coding-agent-learn", "imp-0.8", "embedded-esp32"]
---

记忆正文内容...

---
> 领域: [[_moc/embedded-esp32|embedded-esp32]]
```

## 斜杠命令

| 命令 | 说明 |
|------|------|
| `/memory-sync` | 手动触发双向同步（导出 + 导入 + git） |
| `/memory-export` | 仅导出（Mnemopi → Markdown） |
| `/memory-import` | 仅导入（inbox → Mnemopi） |
| `/memory-recall` | 向量语义检索（参数: 查询文本 [结果数量]） |
| `/memory-config` | 显示当前配置和统计信息 |
| `/memory-domains` | 查看知识领域分布统计 |
| `/memory-cluster` | 手动触发 AI 领域聚类 |

## 事件钩子

| 事件 | 触发条件 | 动作 |
|------|----------|------|
| `turn_start` | 每回合开始 | 主动检索注入相关记忆摘要 |
| `tool_result` | `retain` 或 `learn` 工具完成 | 防抖 3s 后增量导出 + git commit |
| `turn_end` | 每回合结束 | 检查 inbox/ 并导入 |

## 知识领域聚类工作原理

```
记忆写入 → 分类到已有领域 → 累积未分类 ≥ 阈值(30)
                                    ↓
                        embedQuery() 获取所有记忆向量
                                    ↓
                        K-means++ 聚类 (k = √(n/2))
                                    ↓
                        每个聚类提取关键词 → 生成领域
                                    ↓
                        .domain-store.json 持久化
                                    ↓
                        重新分类所有记忆 + 更新 MOC
```

领域发现完全自主，不预设分类规则。新记忆按关键词匹配归入已有领域，无匹配则标为 `unclassified`，等待下次聚类。

## 研究基础

- **A-MEM (Zettelkasten)**: 记忆自主生成关键词/标签/链接
- **MemCog**: proactive reasoning protocol，回合开始主动检索
- **RSC**: recursive semantic chunking，长记忆按段落分块
- **MOC (Map of Content)**: Obsidian 知识图谱聚类 hub 笔记
- **K-means++ clustering**: 嵌入向量无监督聚类发现知识领域

## 依赖

- [Oh My Pi](https://github.com/earendil-works/pi-coding-agent) 编程助手
- [Mnemopi](https://github.com/earendil-works/pi-mnemopi) 记忆系统
- [TypeBox](https://github.com/sinclairzx/typebox) 运行时类型校验
- 嵌入模型 API（默认 Cloudflare Workers AI `bge-m3`）

## License

MIT
