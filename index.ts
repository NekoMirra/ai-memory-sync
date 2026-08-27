/**
 * AI Memory Sync — Mnemopi ↔ Obsidian Vault 双向自动同步扩展
 *
 * 功能:
 *   1. retain / learn 写入后自动增量导出为结构化 Markdown 到 Obsidian vault
 *   2. inbox/ 目录下的笔记自动导入回 Mnemopi（带关键词提取）
 *   3. 导出后自动 git commit + push（3 次重试，push 失败不中断）
 *   4. memory_recall 工具 — 模型可主动调用向量化语义检索
 *   5. 回合开始时自动注入相关记忆摘要（proactive retrieval）
 *   6. 结构化 frontmatter: keywords/tags/domain 便于过滤
 *   7. 知识领域化分类 — AI 自主聚类发现领域（无硬编码），生成 MOC hub 索引笔记
 *   8. 多斜杠命令: /memory-sync, /memory-export, /memory-import, /memory-config, /memory-recall, /memory-domains, /memory-cluster
 *
 * 研究基础:
 *   - A-MEM (Zettelkasten): 记忆自主生成关键词/标签/链接
 *   - MemCog: proactive reasoning protocol，回合开始主动检索
 *   - RSC: recursive semantic chunking，长记忆按段落分块
 *   - MOC (Map of Content): Obsidian 知识图谱聚类 hub 笔记
 *   - K-means++ clustering: 嵌入向量无监督聚类发现知识领域
 *
 * 事件钩子: turn_start (proactive recall) + tool_result (export) + turn_end (import)
 *
 * 配置文件: extensions/ai-memory-sync/config.json
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { execSync } from "node:child_process";
import { Database } from "bun:sqlite";
import { Type } from "typebox";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import { Mnemopi } from "D:/NekoMirra/bun-global/node_modules/@oh-my-pi/pi-mnemopi/src/index.ts";
import { embedQuery } from "D:/NekoMirra/bun-global/node_modules/@oh-my-pi/pi-mnemopi/src/core/embeddings.ts";

// ─── 配置加载 ───
// ─── 动态领域存储 ───

interface DiscoveredDomain {
	name: string;
	keywords: string[];
	memberCount: number;
	createdAt: string;
}

interface DomainStore {
	domains: DiscoveredDomain[];
	lastClusterTime: string | null;
	unclassifiedSinceCluster: number;
	totalAtCluster: number;
}

interface SyncConfig {
	vaultDir: string;
	mnemopiDb: string;
	debounceMs: number;
	gitAutoCommit: boolean;
	gitAutoPush: boolean;
	embeddingModel: string;
	embeddingApiUrl: string;
	embeddingApiKey: string;
	proactiveRecall: boolean;
	proactiveTopK: number;
	domainClusterThreshold: number;
}

const CONFIG_PATH = path.join(import.meta.dir, "config.json");

function loadConfig(): SyncConfig {
	const defaults: SyncConfig = {
		vaultDir: "E:/Document/AI-Memory",
		mnemopiDb: "D:/NekoMirra/omp-data/agent/memories/mnemopi/mnemopi.db",
		debounceMs: 3000,
		gitAutoCommit: true,
		gitAutoPush: true,
		embeddingModel: "@cf/baai/bge-m3",
		embeddingApiUrl: "https://api.cloudflare.com/client/v4/accounts/aba312e2d4b676d176cbd94978480469/ai/v1",
		embeddingApiKey: "",
		proactiveRecall: true,
		proactiveTopK: 5,
		domainClusterThreshold: 30,
	};
	try {
		if (fs.existsSync(CONFIG_PATH)) {
			const raw = JSON.parse(fs.readFileSync(CONFIG_PATH, "utf8"));
			return { ...defaults, ...raw };
		}
	} catch (e: unknown) {
		const msg = e instanceof Error ? e.message : String(e);
		console.error(`[ai-memory-sync] 配置文件加载失败，使用默认值: ${msg}`);
	}
	return defaults;
}

const CFG = loadConfig();

// ─── 路径常量 ───

const BRAIN_MEMORIES = path.join(CFG.vaultDir, "brain", "memories");
const BRAIN_DECISIONS = path.join(CFG.vaultDir, "brain", "decisions");
const BRAIN_FAILURES = path.join(CFG.vaultDir, "brain", "failures");
const BRAIN_PROJECTS = path.join(CFG.vaultDir, "brain", "projects");
const INBOX = path.join(CFG.vaultDir, "inbox");
const IMPORTED = path.join(CFG.vaultDir, "brain", "imported");
const SESSIONS = path.join(CFG.vaultDir, "sessions");

// ─── 防抖 ───

let exportTimer: ReturnType<typeof setTimeout> | null = null;
let lastExportedId: string | null = null;
let initDone = false;
let proactiveContext = "";
let proactiveContextReady = false;

// ─── 工具函数 ───

function slug(str: string, max = 60): string {
	return str
		.replace(/[^a-zA-Z0-9\u4e00-\u9fff\-_]/g, "-")
		.replace(/-{2,}/g, "-")
		.replace(/^-|-$/g, "")
		.slice(0, max) || "unknown";
}

function fmtDate(ts?: string): string {
	if (!ts) return new Date().toISOString().slice(0, 19);
	return ts.replace("T", " ").replace(/\.\d+Z?$/, "").slice(0, 19);
}

function ensureDirs(): void {
	for (const dir of [BRAIN_MEMORIES, BRAIN_DECISIONS, BRAIN_FAILURES, BRAIN_PROJECTS, INBOX, IMPORTED, SESSIONS]) {
		fs.mkdirSync(dir, { recursive: true });
	}
}

// ─── 关键词提取（轻量级，不依赖 LLM） ───

const STOP_WORDS = new Set([
	"the","a","an","is","are","was","were","be","been","being","have","has","had",
	"do","does","did","will","would","could","should","may","might","must","can",
	"this","that","these","those","i","you","he","she","it","we","they","them",
	"their","what","which","who","when","where","why","how","all","each","every",
	"both","few","more","most","other","some","such","no","nor","not","only","own",
	"same","so","than","too","very","just","but","for","and","or","if","then",
	"的","了","是","在","有","和","与","或","但","如果","因为","所以","这","那",
	"一个","一种","可以","需要","应该","没有","不是","已经","可能","的话",
	"to","of","in","on","at","by","from","with","about","into","through","during",
	"before","after","above","below","up","down","out","off","over","under",
]);

function extractKeywords(text: string, max = 8): string[] {
	const words: string[] = [];
	// 英文单词
	const enMatches = text.toLowerCase().match(/[a-z][a-z0-9_-]{2,30}/g) ?? [];
	// 中文词（2-6 字）
	const cnMatches = text.match(/[\u4e00-\u9fff]{2,6}/g) ?? [];

	const freq = new Map<string, number>();
	for (const w of [...enMatches, ...cnMatches]) {
		if (STOP_WORDS.has(w) || w.length < 2) continue;
		freq.set(w, (freq.get(w) ?? 0) + 1);
	}

	// 频率排序 + 长度加权
	return [...freq.entries()]
		.map(([word, count]) => ({ word, score: count * (1 + word.length / 10) }))
		.sort((a, b) => b.score - a.score)
		.slice(0, max)
		.map(e => e.word);
}

// ─── 动态领域分类系统（AI 自主聚类发现，非硬编码） ───

const DOMAIN_STORE_PATH = path.join(import.meta.dir, ".domain-store.json");
const MOC_DIR = path.join(CFG.vaultDir, "_moc");

function loadDomainStore(): DomainStore {
	try {
		if (fs.existsSync(DOMAIN_STORE_PATH)) {
			const raw = JSON.parse(fs.readFileSync(DOMAIN_STORE_PATH, "utf8")) as DomainStore;
			return {
				domains: raw.domains ?? [],
				lastClusterTime: raw.lastClusterTime ?? null,
				unclassifiedSinceCluster: raw.unclassifiedSinceCluster ?? 0,
				totalAtCluster: raw.totalAtCluster ?? 0,
			};
		}
	} catch (e: unknown) {
		const msg = e instanceof Error ? e.message : String(e);
		console.error(`[ai-memory-sync] 领域存储加载失败: ${msg}`);
	}
	return { domains: [], lastClusterTime: null, unclassifiedSinceCluster: 0, totalAtCluster: 0 };
}

function saveDomainStore(store: DomainStore): void {
	try {
		fs.writeFileSync(DOMAIN_STORE_PATH, JSON.stringify(store, null, 2), "utf8");
	} catch (e: unknown) {
		const msg = e instanceof Error ? e.message : String(e);
		console.error(`[ai-memory-sync] 领域存储保存失败: ${msg}`);
	}
}

let domainStore = loadDomainStore();

/** 将记忆归类到已有领域，无匹配则标记 unclassified */
function classifyDomain(source: string, content: string): string {
	if (domainStore.domains.length === 0) return "unclassified";
	const text = `${source} ${content}`.toLowerCase();
	let best = "unclassified";
	let bestScore = 0;
	for (const d of domainStore.domains) {
		let score = 0;
		for (const kw of d.keywords) {
			if (text.includes(kw.toLowerCase())) score += 1;
		}
		if (score > bestScore) { bestScore = score; best = d.name; }
	}
	return bestScore >= 1 ? best : "unclassified";
}

/** 判断是否需要触发重新聚类：未分类累积超过阈值且总量增长显著 */
function shouldCluster(totalMemories: number): boolean {
	if (domainStore.lastClusterTime && Date.now() - new Date(domainStore.lastClusterTime).getTime() < 60000) return false;
	const growth = totalMemories - domainStore.totalAtCluster;
	return domainStore.unclassifiedSinceCluster >= CFG.domainClusterThreshold && growth >= CFG.domainClusterThreshold;
}

/** 调用 Mnemopi 的嵌入 API 获取向量 */
async function getEmbedding(text: string): Promise<number[] | null> {
	try {
		const vec = await embedQuery(text);
		if (!vec) return null;
		return Array.from(vec);
	} catch (e: unknown) {
		const msg = e instanceof Error ? e.message : String(e);
		console.error(`[ai-memory-sync] 嵌入向量获取失败: ${msg}`);
		return null;
	}
}

/** 用嵌入向量对未分类记忆做 K-means 聚类，AI 自主发现领域 */
async function discoverDomains(): Promise<void> {
	const unclassified = sqliteQuery<{ id: string; content: string; source: string }>(
		"SELECT id, content, source FROM working_memory WHERE superseded_by IS NULL",
	);
	if (unclassified.length < CFG.domainClusterThreshold) return;

	console.log(`[ai-memory-sync] 开始领域聚类: ${unclassified.length} 条记忆`);

	// 收集所有记忆的嵌入向量
	const vectors: Map<string, number[]> = new Map();
	for (const row of unclassified) {
		const text = `${row.source} ${row.content}`.slice(0, 500);
		const vec = await getEmbedding(text);
		if (vec) vectors.set(row.id, vec);
	}
	if (vectors.size < CFG.domainClusterThreshold) return;

	// 简易 K-means: 先确定 K，再迭代
	const ids = [...vectors.keys()];
	const mat = ids.map(id => vectors.get(id)!);
	const k = Math.min(Math.max(3, Math.floor(Math.sqrt(ids.length / 2))), 12);
	const clusters = kmeans(mat, k);

	// 为每个聚类提取关键词，生成领域
	const newDomains: DiscoveredDomain[] = [];
	for (let ci = 0; ci < clusters.length; ci++) {
		const members = clusters[ci].map(idx => unclassified[idx]);
		if (members.length < 3) continue; // 太小的聚类跳过
		const allText = members.map(m => `${m.source} ${m.content}`).join(" ");
		const keywords = extractKeywords(allText, 10);
		if (keywords.length < 2) continue;
		const name = keywords[0].toLowerCase().replace(/[^a-z0-9\u4e00-\u9fff]/g, "-").slice(0, 20);
		newDomains.push({
			name,
			keywords,
			memberCount: members.length,
			createdAt: new Date().toISOString(),
		});
		console.log(`[ai-memory-sync] 发现领域: ${name} (${members.length} 条, keywords: ${keywords.slice(0, 5).join(", ")})`);
	}

	if (newDomains.length === 0) return;

	// 合并已有领域（保留人工编辑过的）
	const existingNames = new Set(domainStore.domains.map(d => d.name));
	for (const nd of newDomains) {
		if (!existingNames.has(nd.name)) domainStore.domains.push(nd);
	}
	domainStore.lastClusterTime = new Date().toISOString();
	domainStore.unclassifiedSinceCluster = 0;
	domainStore.totalAtCluster = unclassified.length;
	saveDomainStore(domainStore);

	// 重新分类所有记忆并更新 MOC
	const touched = new Set<string>();
	for (const row of unclassified) {
		const domain = classifyDomain(row.source, row.content);
		if (domain !== "unclassified") touched.add(domain);
	}
	for (const d of touched) updateMocForDomain(d);
	console.log(`[ai-memory-sync] 领域聚类完成: ${newDomains.length} 个新领域, 更新 ${touched.size} 个 MOC`);
}

// ─── K-means 实现（纯 TS，无外部依赖） ───

function kmeans(data: number[][], k: number, maxIter = 20): number[][] {
	if (data.length === 0 || k <= 0) return [];
	const dim = data[0].length;
	// K-means++ 初始化
	const centroids: number[][] = [data[Math.floor(Math.random() * data.length)]];
	while (centroids.length < k) {
		const dists = data.map(p => Math.min(...centroids.map(c => euclid(p, c))));
		const sum = dists.reduce((a, b) => a + b, 0);
		let r = Math.random() * sum, acc = 0;
		for (let i = 0; i < data.length; i++) { acc += dists[i]; if (acc >= r) { centroids.push(data[i]); break; } }
	}
	const assignments = new Array(data.length).fill(0);
	for (let iter = 0; iter < maxIter; iter++) {
		let changed = false;
		for (let i = 0; i < data.length; i++) {
			let best = 0, bestD = Infinity;
			for (let c = 0; c < centroids.length; c++) {
				const d = euclid(data[i], centroids[c]);
				if (d < bestD) { bestD = d; best = c; }
			}
			if (assignments[i] !== best) { assignments[i] = best; changed = true; }
		}
		if (!changed && iter > 0) break;
		// 更新质心
		for (let c = 0; c < k; c++) {
			const members = data.filter((_, i) => assignments[i] === c);
			if (members.length > 0) {
				centroids[c] = members[0].map((_, d) => members.reduce((s, m) => s + m[d], 0) / members.length);
			}
		}
	}
	// 按聚类分组索引
	const result: number[][] = Array.from({ length: k }, () => []);
	for (let i = 0; i < data.length; i++) result[assignments[i]].push(i);
	return result;
}

function euclid(a: number[], b: number[]): number {
	let s = 0;
	for (let i = 0; i < a.length; i++) { const d = a[i] - b[i]; s += d * d; }
	return Math.sqrt(s);
}

// ─── MOC 索引笔记生成 ───

function updateMocForDomain(domain: string): void {
	if (domain === "unclassified" || domain === "general") return;
	fs.mkdirSync(MOC_DIR, { recursive: true });
	const mocPath = path.join(MOC_DIR, `${domain}.md`);
	const domainDir = path.join(BRAIN_MEMORIES, domain);
	if (!fs.existsSync(domainDir)) return;
	const files = fs.readdirSync(domainDir).filter(f => f.endsWith(".md"));
	if (files.length === 0) return;
	const keywords = domainStore.domains.find(d => d.name === domain)?.keywords ?? [];
	const lines = [
		"---",
		`domain: "${domain}"`,
		`member_count: ${files.length}`,
		`keywords: [${keywords.map(k => `"${k}"`).join(", ")}]`,
		`updated: "${new Date().toISOString()}"`,
		"---",
		"",
		`# ${domain}`,
		"",
		`> 本领域共 ${files.length} 条记忆。由 AI 自动聚类发现。`,
		"",
		"## 记忆列表",
		"",
		...files.sort().map(f => `- [[${domain}/${f.replace(/\.md$/, "")}]]`),
		"",
	];
	fs.writeFileSync(mocPath, lines.join("\n"), "utf8");
}

function updateMocIndex(): void {
	fs.mkdirSync(MOC_DIR, { recursive: true });
	const domainDirs = fs.existsSync(BRAIN_MEMORIES) ? fs.readdirSync(BRAIN_MEMORIES) : [];
	const index: Record<string, { count: number; keywords: string[] }> = {};
	for (const dir of domainDirs) {
		if (dir === "imported" || dir === "unclassified") continue;
		const fullDir = path.join(BRAIN_MEMORIES, dir);
		if (!fs.statSync(fullDir).isDirectory()) continue;
		const count = fs.readdirSync(fullDir).filter(f => f.endsWith(".md")).length;
		if (count === 0) continue;
		const keywords = domainStore.domains.find(d => d.name === dir)?.keywords ?? [];
		index[dir] = { count, keywords };
	}
	const indexData = {
		updated: new Date().toISOString(),
		totalDomains: Object.keys(index).length,
		domains: index,
	};
	fs.writeFileSync(path.join(MOC_DIR, "_index.json"), JSON.stringify(indexData, null, 2), "utf8");
}

function getDomainStats(): Record<string, number> {
	const stats: Record<string, number> = {};
	if (!fs.existsSync(BRAIN_MEMORIES)) return stats;
	for (const dir of fs.readdirSync(BRAIN_MEMORIES)) {
		const fullDir = path.join(BRAIN_MEMORIES, dir);
		if (!fs.statSync(fullDir).isDirectory()) continue;
		stats[dir] = fs.readdirSync(fullDir).filter(f => f.endsWith(".md")).length;
	}
	return stats;
}

// ─── 语义分块（长记忆分段，每段可独立检索） ───

function semanticChunks(text: string, maxLen = 500): string[] {
	if (text.length <= maxLen) return [text];
	// 先按段落分，段落过长再按句分
	const paragraphs = text.split(/\n{2,}/);
	const chunks: string[] = [];
	let current = "";
	for (const para of paragraphs) {
		if (para.length > maxLen) {
			if (current) { chunks.push(current); current = ""; }
			const sentences = para.split(/(?<=[.。!！?？\n])/);
			for (const s of sentences) {
				if (current.length + s.length > maxLen && current) {
					chunks.push(current);
					current = s;
				} else {
					current = current ? `${current}${s}` : s;
				}
			}
		} else if (current.length + para.length + 2 > maxLen && current) {
			chunks.push(current);
			current = para;
		} else {
			current = current ? `${current}\n\n${para}` : para;
		}
	}
	if (current) chunks.push(current);
	return chunks.length > 0 ? chunks : [text];
}

// Mnemopi 实例（懒加载，带嵌入配置，走 remember() 触发向量生成）
let _mnemopi: InstanceType<typeof Mnemopi> | null = null;
function mnemopi(): InstanceType<typeof Mnemopi> {
	if (!_mnemopi) {
		_mnemopi = new Mnemopi({
			dbPath: CFG.mnemopiDb,
			sessionId: "ai-memory-sync",
			bank: "default",
			embeddingModel: CFG.embeddingModel,
			embeddingApiUrl: CFG.embeddingApiUrl,
			embeddingApiKey: CFG.embeddingApiKey,
		});
	}
	return _mnemopi;
}

// bun:sqlite 连接（懒加载，FTS5 支持）
let _db: InstanceType<typeof Database> | null = null;
function db(): InstanceType<typeof Database> {
	if (!_db) _db = new Database(CFG.mnemopiDb);
	return _db;
}

function sqliteQuery<T = Record<string, unknown>>(sql: string, params: unknown[] = []): T[] {
	try {
		return db().query(sql).all(...params) as T[];
	} catch (e) {
		console.error("[ai-memory-sync] query 失败:", e);
		return [];
	}
}

// ─── Git 操作 ───

function gitCommit(message: string): boolean {
	if (!CFG.gitAutoCommit) return false;
	const vault = CFG.vaultDir;
	try {
		execSync("git add -A", { cwd: vault, encoding: "utf8", timeout: 10000 });
		const status = execSync("git status --porcelain", { cwd: vault, encoding: "utf8", timeout: 5000 });
		if (!status.trim()) return false;
		execSync(`git commit -m "${message.replace(/"/g, '\\"')}"`, { cwd: vault, encoding: "utf8", timeout: 10000 });
	} catch (e: unknown) {
		const msg = e instanceof Error ? e.message.split("\n")[0] : String(e);
		console.error(`[ai-memory-sync] git commit 失败: ${msg}`);
		return false;
	}
	if (!CFG.gitAutoPush) return true;
	for (let attempt = 1; attempt <= 3; attempt++) {
		try {
			execSync("git push", { cwd: vault, encoding: "utf8", timeout: 30000 });
			return true;
		} catch (e: unknown) {
			const err = e instanceof Error ? e : null;
			const stderr = (err as Error & { stderr?: string })?.stderr;
			const msg = stderr?.split("\n").find((l: string) => l.includes("error") || l.includes("fatal")) ?? err?.message.split("\n")[0] ?? "unknown";
			if (attempt < 3) {
				console.error(`[ai-memory-sync] git push 重试 ${attempt}/3: ${msg}`);
			} else {
				console.error(`[ai-memory-sync] git push 放弃（commit 已成功）: ${msg}`);
			}
		}
	}
	return true;
}

// ─── 导出: 单条记忆 → 结构化 Markdown ───

interface MemoryRow {
	id: string;
	content: string;
	importance: number;
	source: string;
	scope: string;
	timestamp: string;
	metadata_json?: string;
}

function exportMemoryToMarkdown(row: MemoryRow): string {
	const src = row.source || "unknown";
	const domain = classifyDomain(src, row.content || "");
	const domainDir = path.join(BRAIN_MEMORIES, domain);
	fs.mkdirSync(domainDir, { recursive: true });

	const fname = `${fmtDate(row.timestamp).replace(/[: ]/g, "-")}_${String(row.id || "").slice(0, 8)}.md`;
	const fpath = path.join(domainDir, fname);

	const keywords = extractKeywords(row.content || "");
	const tags = [slug(src), `imp-${Math.round((row.importance || 0.5) * 10) / 10}`, domain];

	let meta: Record<string, unknown> = {};
	try { meta = JSON.parse(row.metadata_json || "{}"); } catch {}

	const fm = [
		"---",
		`id: "${row.id || ""}"`,
		`source: "${(row.source || "").replace(/"/g, '\\"')}"`,
		`importance: ${row.importance || 0.5}`,
		`timestamp: "${fmtDate(row.timestamp)}"`,
		`scope: "${row.scope || "global"}"`,
		`domain: "${domain}"`,
		`keywords: [${keywords.map(k => `"${k}"`).join(", ")}]`,
		`tags: [${tags.map(t => `"${t}"`).join(", ")}]`,
		...(meta.migrated ? [`migrated: "${String(meta.migrated)}"`] : []),
		"---",
		"",
		row.content || "",
		"",
		domain !== "unclassified" ? `\n---\n> 领域: [[_moc/${domain}|${domain}]]` : "",
		"",
	].join("\n");

	fs.writeFileSync(fpath, fm, "utf8");
	return domain;
}

// ─── 导出: 增量（只导出 last_id 之后的新记忆）───

function exportNewMemories(): number {
	ensureDirs();
	fs.mkdirSync(MOC_DIR, { recursive: true });
	const touchedDomains = new Set<string>();

	if (!lastExportedId) {
		const rows = sqliteQuery<MemoryRow>(
			"SELECT id, content, importance, source, scope, timestamp, metadata_json FROM working_memory WHERE superseded_by IS NULL ORDER BY timestamp ASC",
		);
		for (const row of rows) {
			const domain = exportMemoryToMarkdown(row);
			touchedDomains.add(domain);
			lastExportedId = row.id;
		}
		console.log(`[ai-memory-sync] 全量导出 ${rows.length} 条记忆`);
		if (rows.length > 0 && CFG.gitAutoCommit) {
			gitCommit(`sync: full export ${rows.length} memories`);
		}
		for (const d of touchedDomains) updateMocForDomain(d);
		updateMocIndex();
		// 触发聚类检查
		const stats = getStats();
		if (shouldCluster(stats.total)) void discoverDomains().catch(e => console.error(`[ai-memory-sync] 聚类失败: ${e instanceof Error ? e.message : String(e)}`));
		return rows.length;
	}

	// 增量: 用 rowid 游标避免 created_at 相同时漏导
	const cursorRows = sqliteQuery<{ rowid: number; id: string }>(
		"SELECT rowid, id FROM working_memory WHERE id = ?",
		[lastExportedId],
	);
	if (cursorRows.length === 0) {
		lastExportedId = null;
		return exportNewMemories();
	}
	const cursorRowid = cursorRows[0].rowid;
	const rows = sqliteQuery<MemoryRow>(
		`SELECT id, content, importance, source, scope, timestamp, metadata_json FROM working_memory WHERE superseded_by IS NULL AND rowid > ? ORDER BY rowid ASC`,
		[cursorRowid],
	);
	for (const row of rows) {
		const domain = exportMemoryToMarkdown(row);
		touchedDomains.add(domain);
		lastExportedId = row.id;
		if (domain === "unclassified") domainStore.unclassifiedSinceCluster++;
	}
	if (rows.length > 0) {
		console.log(`[ai-memory-sync] 增量导出 ${rows.length} 条记忆`);
		if (CFG.gitAutoCommit) {
			gitCommit(`sync: ${rows.length} new memories`);
		}
	}
	for (const d of touchedDomains) updateMocForDomain(d);
	updateMocIndex();
	// 触发聚类检查
	const stats = getStats();
	if (shouldCluster(stats.total)) {
		saveDomainStore(domainStore);
		void discoverDomains().catch(e => console.error(`[ai-memory-sync] 聚类失败: ${e instanceof Error ? e.message : String(e)}`));
	}
	return rows.length;
}

// ─── 导入: inbox/ → Mnemopi（带分块） ───

function importInbox(): number {
	if (!fs.existsSync(INBOX)) return 0;

	const files = fs.readdirSync(INBOX).filter(f => f.endsWith(".md"));
	let imported = 0;

	for (const file of files) {
		const fpath = path.join(INBOX, file);
		const raw = fs.readFileSync(fpath, "utf8");

		let content = raw;
		let source = `obsidian/${file}`;
		let importance = 0.7;
		const fmMatch = raw.match(/^---\n([\s\S]*?)\n---\n([\s\S]*)/);
		if (fmMatch) {
			content = fmMatch[2].trim();
			for (const line of fmMatch[1].split("\n")) {
				const sm = line.match(/^source:\s*"?([^"\n]*)"?\s*$/);
				if (sm) source = sm[1];
				const im = line.match(/^importance:\s*([\d.]+)/);
				if (im) importance = parseFloat(im[1]);
			}
		}

		if (!content.trim()) continue;

		// 语义分块后逐块写入（长笔记分成多条记忆）
		const chunks = semanticChunks(content);
		let chunkErrors = 0;
		for (const chunk of chunks) {
			try {
				const mem = mnemopi();
				const id = mem.remember(chunk, { source, importance });
				console.log(`[ai-memory-sync] remember() OK: ${id} (source=${source})`);
				imported++;
			} catch (e) {
				chunkErrors++;
				const msg = e instanceof Error ? e.message.split("\n")[0] : String(e);
				console.error(`[ai-memory-sync] 导入 ${file} 块失败: ${msg}`);
			}
		}

		// 所有块都成功才移到 imported，否则留在 inbox 下次重试
		if (chunkErrors === 0) {
			fs.mkdirSync(IMPORTED, { recursive: true });
			fs.renameSync(fpath, path.join(IMPORTED, file));
		} else {
			console.error(`[ai-memory-sync] ${file} 有 ${chunkErrors} 块失败，保留在 inbox`);
		}
	}

	if (imported > 0) {
		console.log(`[ai-memory-sync] 从 inbox 导入 ${imported} 条`);
		if (CFG.gitAutoCommit) {
			gitCommit(`sync: imported ${imported} from inbox`);
		}
	}
	return imported;
}

// ─── 向量检索 ───

async function vectorRecall(query: string, topK = 5): Promise<RecallResult[]> {
	try {
		return await mnemopi().recall(query, topK);
	} catch (e) {
		const msg = e instanceof Error ? e.message : String(e);
		console.error(`[ai-memory-sync] 向量检索失败: ${msg}`);
		return [];
	}
}

// ─── 主动检索: 回合开始时注入相关记忆 ───

function buildProactiveContext(query: string): void {
	if (!CFG.proactiveRecall) return;
	void vectorRecall(query, CFG.proactiveTopK).then(results => {
		if (results.length === 0) {
			proactiveContextReady = false;
			return;
		}
		const lines = results.map((r, i) => {
			const score = r.score ? r.score.toFixed(2) : "?";
			const content = (r.content ?? "").slice(0, 120);
			return `${i + 1}. [${score}] ${content}${r.truncated ? "…" : ""}`;
		});
		proactiveContext = `[相关记忆 ${results.length} 条]\n${lines.join("\n")}`;
		proactiveContextReady = true;
		console.log(`[ai-memory-sync] 主动检索注入 ${results.length} 条记忆`);
	}).catch(() => { proactiveContextReady = false; });
}

// ─── 防抖导出 ───

function scheduleExport(): void {
	if (exportTimer) clearTimeout(exportTimer);
	exportTimer = setTimeout(() => {
		exportTimer = null;
		try {
			exportNewMemories();
		} catch (e) {
			const msg = e instanceof Error ? e.message : String(e);
			console.error(`[ai-memory-sync] 导出失败: ${msg}`);
		}
	}, CFG.debounceMs);
}

// ─── 统计 ───

function getStats(): { total: number; lastExport: string | null } {
	const rows = sqliteQuery<{ cnt: number }>("SELECT COUNT(*) as cnt FROM working_memory WHERE superseded_by IS NULL");
	const total = rows[0]?.cnt ?? 0;
	const lastRow = sqliteQuery<{ timestamp: string }>("SELECT timestamp FROM working_memory WHERE superseded_by IS NULL ORDER BY timestamp DESC LIMIT 1");
	const lastExport = lastRow[0]?.timestamp ?? null;
	return { total, lastExport };
}

// ─── 扩展入口 ───

export default function (pi: ExtensionAPI): void {
	ensureDirs();

	// 初始化: 全量导出一次
	try {
		const count = exportNewMemories();
		initDone = true;
		console.log(`[ai-memory-sync] 初始化完成，导出 ${count} 条记忆到 ${CFG.vaultDir}`);
	} catch (e) {
		const msg = e instanceof Error ? e.message : String(e);
		console.error(`[ai-memory-sync] 初始化失败: ${msg}`);
	}

	// 钩子 1: retain / learn 工具完成后 → 防抖导出
	pi.on("tool_result", (event: Record<string, unknown>, _ctx: unknown) => {
		if (event.type !== "tool_result") return;
		const toolName = event.toolName as string;
		if (toolName !== "retain" && toolName !== "learn") return;

		const input = event.input as Record<string, unknown> | undefined;
		if (input?.items) {
			const n = Array.isArray(input.items) ? input.items.length : 1;
			console.log(`[ai-memory-sync] 检测到 ${toolName} 写入 ${n} 条，触发同步`);
		} else if (input?.memory) {
			console.log(`[ai-memory-sync] 检测到 ${toolName} 写入，触发同步`);
		}

		scheduleExport();
	});

	// 钩子 2: 回合开始 → 主动检索注入相关记忆
	pi.on("turn_start", (event: Record<string, unknown>, _ctx: unknown) => {
		try {
			const userMsg = event.userMessage ?? event.message ?? "";
			if (typeof userMsg === "string" && userMsg.trim()) {
				buildProactiveContext(userMsg.slice(0, 500));
			}
		} catch { /* 主动检索失败不影响主流程 */ }
	});

	// 钩子 3: 每回合结束 → 检查 inbox 导入（仅在 inbox 有 .md 文件时）
	pi.on("turn_end", (_event: unknown, _ctx: unknown) => {
		try {
			if (fs.existsSync(INBOX)) {
				const files = fs.readdirSync(INBOX);
				if (files.some(f => f.endsWith(".md"))) {
					importInbox();
				}
			}
		} catch (e) {
			const msg = e instanceof Error ? e.message : String(e);
			console.error(`[ai-memory-sync] inbox 导入失败: ${msg}`);
		}
	});

	// 工具: memory_recall — 模型主动调用向量化语义检索
	pi.registerTool({
		name: "memory_recall",
		label: "Memory Recall",
		description: "Search long-term memory via vector similarity. Returns top-K relevant memories with scores. Use proactively before answering questions about past conversations, project history, or user preferences.",
		promptSnippet: "Recall relevant memories from the vector store.",
		promptGuidelines: [
			"Use memory_recall before answering questions about past sessions, project decisions, user preferences, or debugging history.",
			"Use memory_recall when the user asks 'do you remember' or references prior work.",
			"Queries should be natural language phrases describing what you're looking for.",
		],
		parameters: Type.Object({
			query: Type.String({ minLength: 1, description: "Natural language search query for semantic memory retrieval." }),
			top_k: Type.Optional(Type.Integer({ minimum: 1, maximum: 20, default: 5, description: "Number of results to return (default 5)." }),
			),
		}),
		async execute(_id, params) {
			const q = String(params.query ?? "");
			const topK = Math.min(20, Math.max(1, Number(params.top_k) || 5));
			const results = await vectorRecall(q, topK);
			if (results.length === 0) {
				return { content: [{ type: "text", text: "No matching memories found." }], details: { count: 0 } };
			}
			const lines = results.map((r, i) => {
				const score = r.score ? r.score.toFixed(3) : "?";
				const dist = r.distance ? ` dist=${r.distance.toFixed(3)}` : "";
				const content = (r.content ?? "").slice(0, 300);
				const preview = r.truncated ? `${content}… (${r.full_length} chars total)` : content;
				const src = r.source ? ` source=${r.source}` : "";
				const imp = r.importance != null ? ` imp=${r.importance}` : "";
				return `## ${i + 1}. [score=${score}${dist}]${src}${imp}\n${preview}\n`;
			});
			const text = lines.join("\n");
			return {
				content: [{ type: "text", text }],
				details: { count: results.length, query: q, topK },
			};
		},
	});

	// 命令: /memory-sync — 手动双向同步
	pi.registerCommand("memory-sync", {
		description: "手动触发 Mnemopi ↔ Obsidian vault 双向同步（导出 + 导入 + git commit）",
		handler: async (ctx) => {
			const imported = importInbox();
			const exported = exportNewMemories();
			const stats = getStats();
			const msg = `同步完成: 导入 ${imported} 条, 导出 ${exported} 条, 总计 ${stats.total} 条记忆`;
			console.log(`[ai-memory-sync] ${msg}`);
			ctx.ui?.notice?.(msg);
		},
	});

	// 命令: /memory-export — 仅导出
	pi.registerCommand("memory-export", {
		description: "手动导出 Mnemopi 记忆到 Obsidian vault（单向: Mnemopi → Markdown）",
		handler: async (ctx) => {
			const exported = exportNewMemories();
			const msg = `导出完成: ${exported} 条记忆`;
			console.log(`[ai-memory-sync] ${msg}`);
			ctx.ui?.notice?.(msg);
		},
	});

	// 命令: /memory-import — 仅导入
	pi.registerCommand("memory-import", {
		description: "手动导入 Obsidian inbox 中的笔记到 Mnemopi（单向: Markdown → Mnemopi）",
		handler: async (ctx) => {
			const imported = importInbox();
			const msg = `导入完成: ${imported} 条`;
			console.log(`[ai-memory-sync] ${msg}`);
			ctx.ui?.notice?.(msg);
		},
	});

	// 命令: /memory-recall — 手动向量检索
	pi.registerCommand("memory-recall", {
		description: "向量语义检索记忆（参数: 查询文本 [结果数量]）",
		handler: async (args, ctx) => {
			const parts = String(args ?? "").trim().split(/\s+/);
			const topK = /^\d+$/.test(parts[parts.length - 1] ?? "") ? parseInt(parts.pop()!, 10) : 5;
			const query = parts.join(" ");
			if (!query) {
				ctx.ui?.notice?.("用法: /memory-recall <查询文本> [结果数量]");
				return;
			}
			const results = await vectorRecall(query, topK);
			if (results.length === 0) {
				ctx.ui?.notice?.("未找到匹配的记忆");
				return;
			}
			const lines = results.map((r, i) => {
				const score = r.score ? r.score.toFixed(3) : "?";
				const content = (r.content ?? "").slice(0, 150);
				return `${i + 1}. [${score}] ${content}${r.truncated ? "…" : ""}`;
			});
			const msg = lines.join("\n");
			console.log(`[ai-memory-sync] 检索 "${query}" → ${results.length} 条`);
			ctx.ui?.notice?.(msg);
		},
	});

	// 命令: /memory-config — 显示配置
	pi.registerCommand("memory-config", {
		description: "显示 AI Memory Sync 扩展的当前配置和统计信息",
		handler: async (ctx) => {
			const stats = getStats();
			const lines = [
				"AI Memory Sync 配置:",
				`  Vault 目录: ${CFG.vaultDir}`,
				`  Mnemopi DB: ${CFG.mnemopiDb}`,
				`  防抖延迟: ${CFG.debounceMs}ms`,
				`  Git 自动提交: ${CFG.gitAutoCommit ? "开" : "关"}`,
				`  Git 自动推送: ${CFG.gitAutoPush ? "开" : "关"}`,
				`  主动检索: ${CFG.proactiveRecall ? "开" : "关"}`,
				`  主动检索 TopK: ${CFG.proactiveTopK}`,
				`  嵌入模型: ${CFG.embeddingModel}`,
				"",
				"统计:",
				`  Mnemopi 总记忆: ${stats.total} 条`,
				`  最新记忆时间: ${stats.lastExport || "无"}`,
				`  初始化状态: ${initDone ? "完成" : "未完成"}`,
			`  主动检索上下文: ${proactiveContext ? "已加载" : "空"}`,
			`  领域聚类阈值: ${CFG.domainClusterThreshold} 条`,
			`  已发现领域: ${domainStore.domains.length} 个`,
			];
			const msg = lines.join("\n");
			console.log(`[ai-memory-sync]\n${msg}`);
			ctx.ui?.notice?.(msg);
		},
	});

	// 命令: /memory-domains — 查看领域分布统计
	pi.registerCommand("memory-domains", {
		description: "查看知识领域分布统计和 AI 发现的领域",
		handler: async (ctx) => {
			const stats = getDomainStats();
	const total = Object.values(stats).reduce((a, b) => a + b, 0);
			const lines = [
				"知识领域分布:",
				`  总记忆: ${total} 条`,
				`  已发现领域: ${domainStore.domains.length} 个`,
				`  聚类阈值: ${CFG.domainClusterThreshold} 条`,
				`  上次聚类: ${domainStore.lastClusterTime ?? "从未"}`,
				`  待聚类未分类: ${domainStore.unclassifiedSinceCluster} 条`,
				"",
				"领域列表:",
				...Object.entries(stats)
					.sort(([, a], [, b]) => b - a)
					.map(([name, count]) => `  ${name}: ${count} 条 (${Math.round(count / total * 100)}%)`),
			];
			if (domainStore.domains.length > 0) {
				lines.push("", "AI 发现的领域关键词:");
				for (const d of domainStore.domains) {
					lines.push(`  ${d.name} (${d.memberCount} 条): ${d.keywords.slice(0, 5).join(", ")}`);
				}
			}
			const msg = lines.join("\n");
			console.log(`[ai-memory-sync]\n${msg}`);
			ctx.ui?.notice?.(msg);
		},
	});

	// 命令: /memory-cluster — 手动触发领域聚类
	pi.registerCommand("memory-cluster", {
		description: "手动触发 AI 领域聚类（用嵌入向量对所有记忆做 K-means 聚类）",
		handler: async (ctx) => {
			ctx.ui?.notice?.("开始领域聚类，请稍候...");
			try {
				await discoverDomains();
				const msg = domainStore.domains.length > 0
					? `聚类完成: 发现 ${domainStore.domains.length} 个领域`
					: "聚类完成: 记忆数量不足或无法生成领域";
				console.log(`[ai-memory-sync] ${msg}`);
				ctx.ui?.notice?.(msg);
			} catch (e: unknown) {
				const msg = e instanceof Error ? e.message : String(e);
				console.error(`[ai-memory-sync] 聚类失败: ${msg}`);
				ctx.ui?.notice?.(`聚类失败: ${msg}`);
			}
		},
	});
}
