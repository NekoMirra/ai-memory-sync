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
 *   9. 记忆去重合并 — remember 前先 recall 查语义相似度，>0.85 合并而非新增
 *  10. importance 动态衰减 — recall 命中升权，长期未命中降权
 *  11. transcript 摘要提取 — 对话原文不直存，提取核心信息
 *  12. 长记忆强制分块 — >1000 字符的记忆在导出时按语义分块
 *
 * 研究基础:
 *   - A-MEM (Zettelkasten): 记忆自主生成关键词/标签/链接
 *   - MemCog: proactive reasoning protocol，回合开始主动检索
 *   - RSC: recursive semantic chunking，长记忆按段落分块
 *   - MOC (Map of Content): Obsidian 知识图谱聚类 hub 笔记
 *   - K-means++ clustering: 嵌入向量无监督聚类发现知识领域
 *   - Forgetting curve: recall 频率驱动的 importance 动态衰减
 *
 * 事件钩子: before_agent_start (proactive recall) + tool_result (export) + turn_end (import + decay)
 *
 * 配置文件: extensions/ai-memory-sync/config.json
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { execSync } from "node:child_process";
import { Database } from "bun:sqlite";
import { Type } from "typebox";
import type { ExtensionAPI, ExtensionContext } from "@earendil-works/pi-coding-agent";
import { Mnemopi } from "@oh-my-pi/pi-mnemopi";
import type { RecallResult } from "@oh-my-pi/pi-mnemopi/types";

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
	dedupThreshold: number;
	decayEnabled: boolean;
	decayInterval: number;
	chunkThreshold: number;
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
		dedupThreshold: 0.85,
		decayEnabled: true,
		decayInterval: 3600,
		chunkThreshold: 1000,
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

// 密钥占位符（$$…$$）由 omp 在进程启动时还原进环境变量，扩展读 config.json 拿到的是字面量
const EMBED_KEY = /^\$\$/.test(CFG.embeddingApiKey)
	? (process.env.CLOUDFLARE_API_TOKEN ?? "")
	: CFG.embeddingApiKey;
if (EMBED_KEY === "") {
	console.error("[ai-memory-sync] 未取得 embedding API key：config.json 的 embeddingApiKey 是占位符且 process.env.CLOUDFLARE_API_TOKEN 为空");
}

// ─── 路径常量 ───

const BRAIN_MEMORIES = path.join(CFG.vaultDir, "brain", "memories");
const BRAIN_DECISIONS = path.join(CFG.vaultDir, "brain", "decisions");
const BRAIN_FAILURES = path.join(CFG.vaultDir, "brain", "failures");
const BRAIN_PROJECTS = path.join(CFG.vaultDir, "brain", "projects");
const INBOX = path.join(CFG.vaultDir, "inbox");
const IMPORTED = path.join(CFG.vaultDir, "brain", "imported");
const SESSIONS = path.join(CFG.vaultDir, "sessions");

// ─── 防抖 ───

let exportTimer: Timer | null = null;
let lastExportedId: string | null = null;
let initDone = false;
let lastDecayTime = 0;

// 导出游标：按 rowid 递增推进，落盘避免每次会话重复全量导出
const CURSOR_PATH = path.join(import.meta.dir, ".sync-cursor.json");

/** 读游标文件，不存在或损坏视为无游标 */
function loadCursor(): string | null {
	try {
		if (!fs.existsSync(CURSOR_PATH)) return null;
		const raw = JSON.parse(fs.readFileSync(CURSOR_PATH, "utf8")) as { lastExportedId?: string | null };
		return raw.lastExportedId ?? null;
	} catch (e: unknown) {
		console.error(`[ai-memory-sync] 游标读取失败，按无游标处理: ${e instanceof Error ? e.message : String(e)}`);
		return null;
	}
}

/** 落盘当前游标；safe for concurrent calls（单文件整体覆盖写） */
function saveCursor(): void {
	try {
		fs.writeFileSync(CURSOR_PATH, JSON.stringify({ lastExportedId }, null, 2), "utf8");
	} catch (e: unknown) {
		console.error(`[ai-memory-sync] 游标保存失败: ${e instanceof Error ? e.message : String(e)}`);
	}
}

lastExportedId = loadCursor();

// ─── 工具函数 ───

function slug(str: string, max = 60): string {
	return str
		.replace(/[^a-zA-Z0-9\u4e00-\u9fff\-_]/g, "-")
		.replace(/-{2,}/g, "-")
		.replace(/^-|-$/g, "")
		.slice(0, max) || "unknown";
}

/** 规范化时间戳为 "YYYY-MM-DD HH:MM:SS"；无值返回空串，不回落当前时间 */
function fmtDate(ts?: string): string {
	if (!ts) return "";
	return ts.replace("T", " ").replace(/\.\d+Z?$/, "").slice(0, 19);
}

/** 笔记文件名：时间戳 + 记忆 id 前 8 位；同一记忆的文件名恒定 */
function memoryFileName(id: string, stamp: string): string {
	return `${stamp.replace(/[^0-9A-Za-z]+/g, "-")}_${id.slice(0, 8)}.md`;
}

/** 记忆的导出时间戳：优先记录时间，其次入库时间，两者皆空标为 undated */
function stampOf(row: Pick<MemoryRow, "timestamp" | "created_at">): string {
	return fmtDate(row.timestamp || row.created_at) || "undated";
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

/** 走宿主同一模型的批量嵌入；safe for concurrent calls（只读 CFG/EMBED_KEY + fetch） */
async function embedBatch(texts: string[]): Promise<(number[] | null)[]> {
	if (texts.length === 0) return [];
	try {
		const res = await fetch(`${CFG.embeddingApiUrl.replace(/\/+$/, "")}/embeddings`, {
			method: "POST",
			headers: { "Content-Type": "application/json", Authorization: `Bearer ${EMBED_KEY}` },
			body: JSON.stringify({ model: CFG.embeddingModel, input: texts }),
			signal: AbortSignal.timeout(30000),
		});
		if (!res.ok) {
			console.error(`[ai-memory-sync] 嵌入请求失败: HTTP ${res.status}`);
			return texts.map(() => null);
		}
		const payload = (await res.json()) as { data?: Array<{ embedding: number[] }> };
		const data = payload.data;
		if (!data) return texts.map(() => null);
		return texts.map((_, i) => data[i]?.embedding ?? null);
	} catch (e: unknown) {
		console.error(`[ai-memory-sync] 嵌入请求异常: ${e instanceof Error ? e.message : String(e)}`);
		return texts.map(() => null);
	}
}

/** 用嵌入向量对未分类记忆做 K-means 聚类，AI 自主发现领域 */
async function discoverDomains(): Promise<void> {
	const unclassified = sqliteQuery<{ id: string; content: string; source: string }>(
		"SELECT id, content, source FROM working_memory WHERE superseded_by IS NULL",
	);
	if (unclassified.length < CFG.domainClusterThreshold) return;

	console.log(`[ai-memory-sync] 开始领域聚类: ${unclassified.length} 条记忆`);

	// 优先复用库内已有的 bge-m3 向量，只对缺向量的行批量补算
	const stored = new Map<string, number[]>();
	for (const row of sqliteQuery<{ memory_id: string; embedding_json: string }>(
		"SELECT memory_id, embedding_json FROM memory_embeddings",
	)) {
		try {
			const vec = JSON.parse(row.embedding_json) as number[];
			if (Array.isArray(vec) && vec.length > 0) stored.set(row.memory_id, vec);
		} catch {
			console.error(`[ai-memory-sync] 跳过坏向量行 ${row.memory_id}`);
		}
	}
	const wanted = new Set(unclassified.map(r => r.id));
	const vectors = new Map<string, number[]>();
	for (const [id, vec] of stored) if (wanted.has(id)) vectors.set(id, vec);
	const missing = unclassified.filter(r => !vectors.has(r.id));
	for (let i = 0; i < missing.length; i += 32) {
		const slice = missing.slice(i, i + 32);
		const vecs = await embedBatch(slice.map(r => `${r.source ?? ""} ${r.content}`.slice(0, 500)));
		vecs.forEach((vec, j) => { if (vec) vectors.set(slice[j].id, vec); });
	}
	if (vectors.size < CFG.domainClusterThreshold) return;

	// 简易 K-means: 先确定 K，再迭代
	const ids = [...vectors.keys()];
	const mat = ids.map(id => vectors.get(id)!);
	const k = Math.min(Math.max(3, Math.floor(Math.sqrt(ids.length / 2))), 12);
	const clusters = kmeans(mat, k);

	// 为每个聚类提取关键词，生成领域
	const byId = new Map(unclassified.map(r => [r.id, r]));
	const newDomains: DiscoveredDomain[] = [];
	for (let ci = 0; ci < clusters.length; ci++) {
		const members = clusters[ci]
			.map(idx => byId.get(ids[idx]!))
			.filter((r): r is typeof unclassified[number] => r !== undefined);
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

	// 重新分类所有记忆并更新 MOC（聚类在 gitCommit 之后异步跑完，这里改动的文件由本次提交兜住）
	const touched = new Set<string>();
	for (const row of unclassified) {
		const domain = classifyDomain(row.source, row.content);
		if (domain !== "unclassified") touched.add(domain);
	}
	let mocChanged = false;
	for (const d of touched) mocChanged = updateMocForDomain(d) || mocChanged;
	mocChanged = updateMocIndex() || mocChanged;
	if (mocChanged) gitCommit("sync: 领域聚类刷新 MOC");
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

/** 内容未变化时跳过写入；比较时忽略只表示刷新时间的 updated 字段，避免每次启动产生无意义提交 */
function writeIfChanged(filePath: string, content: string): boolean {
	if (fs.existsSync(filePath)) {
		const stripRefreshTime = (s: string) => s.split("\n").filter(l => !/^\s*"?updated"?\s*:/.test(l)).join("\n");
		const prev = fs.readFileSync(filePath, "utf8");
		if (prev === content || stripRefreshTime(prev) === stripRefreshTime(content)) return false;
	}
	fs.writeFileSync(filePath, content, "utf8");
	return true;
}

/** 写领域 MOC；返回文件是否真的变化 */
function updateMocForDomain(domain: string): boolean {
	if (domain === "unclassified" || domain === "general") return false;
	fs.mkdirSync(MOC_DIR, { recursive: true });
	const mocPath = path.join(MOC_DIR, `${domain}.md`);
	const domainDir = path.join(BRAIN_MEMORIES, domain);
	if (!fs.existsSync(domainDir)) return false;
	const files = fs.readdirSync(domainDir).filter(f => f.endsWith(".md"));
	if (files.length === 0) return false;
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
	return writeIfChanged(mocPath, lines.join("\n"));
}

/** 写 MOC 索引；返回文件是否真的变化 */
function updateMocIndex(): boolean {
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
	return writeIfChanged(path.join(MOC_DIR, "_index.json"), JSON.stringify(indexData, null, 2));
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

// ─── [新] 记忆去重合并 ───
// remember 前先 recall 查语义相似度，>阈值则合并而非新增

/**
 * 检查新记忆是否和已有记忆高度相似。
 * 如果相似度 > dedupThreshold，合并到已有记忆（更新 content + 提升 importance）。
 * @returns 合并到的 memoryId，或 null 表示没有重复（应新建）
 */
async function dedupCheck(content: string, source: string): Promise<string | null> {
	if (content.trim().length < 20) return null;
	try {
		const results = await mnemopi().recall(content.slice(0, 500), 3);
		for (const r of results) {
			if (r.score != null && r.score >= CFG.dedupThreshold) {
				// 合并：用新内容更新已有记忆（保留较长版本），importance 取 max
				const existingContent = r.content ?? "";
				const merged = content.length > existingContent.length ? content : existingContent;
				const existingImp = r.importance ?? 0.5;
				const newImp = Math.min(1.0, Math.max(existingImp, 0.75) + 0.05);
				mnemopi().update(r.id, merged, newImp);
				console.log(`[ai-memory-sync] 记忆去重合并: ${r.id} (score=${r.score.toFixed(3)}, src=${source}→${r.source ?? "?"})`);
				return r.id;
			}
		}
	} catch {
		// 去重检查失败不阻断写入
	}
	return null;
}

/** 带去重的 remember：先查重，有重复则合并，无则新建 */
async function dedupRemember(content: string, source: string, importance: number): Promise<string> {
	const existingId = await dedupCheck(content, source);
	if (existingId) return existingId;
	return mnemopi().remember(content, { source, importance });
}

// ─── [新] importance 动态衰减 ───
// recall_count 高的记忆升权，长期 0 命中的降权

/** 对所有记忆执行 importance 衰减/升权，间隔 decayInterval 秒执行一次。
 * 一次查询取全量，逐行走库 API 更新（FTS 与向量随库写入同步） */
function decayImportance(): void {
	if (!CFG.decayEnabled) return;
	const now = Date.now();
	if (now - lastDecayTime < CFG.decayInterval * 1000) return;
	lastDecayTime = now;

	const rows = sqliteQuery<{
		id: string;
		recall_count: number | null;
		importance: number;
		timestamp: string | null;
		created_at: string | null;
	}>("SELECT id, recall_count, importance, timestamp, created_at FROM working_memory WHERE superseded_by IS NULL");

	const cutoff = now - 86400000;
	for (const r of rows) {
		const rc = r.recall_count ?? 0;
		let next: number | null = null;
		// recall_count > 3 升权（+0.02，上限 1.0）
		if (rc > 3 && r.importance < 1.0) next = Math.min(1.0, r.importance + 0.02);
		// recall_count = 0 且超过 24h 降权（-0.01，下限 0.1）
		else if (rc === 0 && r.importance > 0.1 && new Date(r.timestamp || r.created_at || 0).getTime() < cutoff)
			next = Math.max(0.1, r.importance - 0.01);
		// recall_count 1-3 轻微升权（+0.01，上限 0.95）
		else if (rc >= 1 && rc <= 3 && r.importance < 0.95)
			next = Math.min(0.95, r.importance + 0.01);
		if (next === null) continue;
		mnemopi().update(r.id, null, next);
	}
}

// ─── [新] transcript 摘要提取 ───
// 对话原文不直存，提取核心信息

/** 判断是否是 transcript 原文 */
function isTranscript(content: string): boolean {
	return content.includes("[role: user]") || content.includes("[role: assistant]") || content.includes("[user:end]");
}

/**
 * 从 transcript 原文中提取核心摘要。
 * 策略: 过滤无意义消息（继续/jixu/好等），提取有信息量的 user 消息和 assistant 关键回复。
 * 如果提取后太短（<30字符），说明是无价值对话，返回 null。
 */
const NOISE_MSGS = new Set([
	"继续", "jixu", "好", "ok", "嗯", "是", "对", "行", "可以", "收到", "继续吧",
	"继续，记得先网络检索", "继续干活", "继续，记得先网络检索",
]);

/** 判断单条消息是否有信息量 */
function isMeaningfulMsg(msg: string): boolean {
	const trimmed = msg.trim().toLowerCase();
	if (trimmed.length < 5) return false;
	if (NOISE_MSGS.has(trimmed)) return false;
	if (trimmed === "继续" || trimmed === "jixu") return false;
	// 纯标点/表情
	if (!/[a-z\u4e00-\u9fff]/i.test(trimmed)) return false;
	return true;
}

function summarizeTranscript(content: string): string | null {
	if (!isTranscript(content)) return content;

	// 提取所有有信息量的 user 消息
	const userMsgs: string[] = [];
	const userPattern = /\[role:\s*user\]\s*([\s\S]*?)\s*\[user:end\]/g;
	let m: RegExpExecArray | null;
	while ((m = userPattern.exec(content)) !== null) {
		const msg = m[1].trim();
		if (isMeaningfulMsg(msg)) userMsgs.push(msg.slice(0, 200));
	}

	// 提取 assistant 消息的第一个实质段落（跳过 tool call 块）
	const assistantMsgs: string[] = [];
	const asstPattern = /\[role:\s*assistant\]\s*([\s\S]*?)\s*\[assistant:end\]/g;
	while ((m = asstPattern.exec(content)) !== null) {
		const msg = m[1].trim();
		if (msg.length > 50 && !msg.startsWith("<parameter") && !msg.startsWith("```")) {
			assistantMsgs.push(msg.slice(0, 200));
		}
	}

	// 去重 user 消息（同一消息重复多次只保留一次）
	const uniqueUserMsgs = [...new Set(userMsgs)];

	// 拼接摘要
	const parts: string[] = [];
	if (uniqueUserMsgs.length > 0) parts.push(`用户: ${uniqueUserMsgs.join(" / ")}`);
	if (assistantMsgs.length > 0) parts.push(`助手: ${assistantMsgs.slice(0, 3).join(" / ")}`);

	const summary = parts.join("\n").trim();
	if (summary.length < 30) return null;
	return summary.slice(0, 500);
}

// ─── Mnemopi 实例（懒加载，带嵌入配置，走 remember() 触发向量生成） ───

// 与宿主一致：sessionId 必须取 bank 名，否则 recall 的 `session_id = ? OR scope = 'global'` 过滤会把全部记忆挡掉
const MEMORY_BANK = "default";

let _mnemopi: Mnemopi | null = null;
/** 懒加载的 Mnemopi 实例；safe for concurrent calls */
function mnemopi(): Mnemopi {
	if (!_mnemopi) {
		_mnemopi = new Mnemopi({
			dbPath: CFG.mnemopiDb,
			sessionId: MEMORY_BANK,
			bank: MEMORY_BANK,
			embeddingModel: CFG.embeddingModel,
			embeddingApiUrl: CFG.embeddingApiUrl,
			embeddingApiKey: EMBED_KEY,
		});
	}
	return _mnemopi;
}

// bun:sqlite 连接（懒加载，FTS5 支持）
let _db: Database | null = null;
/** 只读句柄；safe for concurrent calls */
function readDb(): Database {
	if (!_db) {
		_db = new Database(CFG.mnemopiDb, { readonly: true });
		_db.run("PRAGMA busy_timeout = 5000");
	}
	return _db;
}

function sqliteQuery<T = Record<string, unknown>>(sql: string, params: unknown[] = []): T[] {
	try {
		return readDb().query(sql).all(...params) as T[];
	} catch (e: unknown) {
		console.error(`[ai-memory-sync] SQL 失败: ${e instanceof Error ? e.message : String(e)} | ${sql}`);
		return [];
	}
}

/** 短命写连接：仅 /memory-clean 标记无价值记忆时使用，用完立即关闭 */
function withWriteDb<T>(fn: (db: Database) => T): T {
	const handle = new Database(CFG.mnemopiDb);
	try {
		handle.run("PRAGMA busy_timeout = 5000");
		return fn(handle);
	} finally {
		handle.close();
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
				try {
					execSync("git pull --rebase origin master", { cwd: vault, encoding: "utf8", timeout: 30000 });
				} catch {
					try {
						execSync("git rebase --abort", { cwd: vault, encoding: "utf8", timeout: 10000 });
					} catch {}
				}
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
	timestamp: string | null;
	created_at: string | null;
	metadata_json?: string;
}

function exportMemoryToMarkdown(row: MemoryRow): string {
	const src = row.source || "unknown";
	let content = row.content || "";

	// transcript 摘要提取
	if (isTranscript(content)) {
		const summary = summarizeTranscript(content);
		if (summary === null) return "unclassified"; // 无价值对话跳过
		content = summary;
	}

	const domain = classifyDomain(src, content);
	const domainDir = path.join(BRAIN_MEMORIES, domain);
	fs.mkdirSync(domainDir, { recursive: true });

	const fname = memoryFileName(String(row.id || ""), stampOf(row));
	const fpath = path.join(domainDir, fname);

	const keywords = extractKeywords(content);
	const tags = [slug(src), `imp-${Math.round((row.importance || 0.5) * 10) / 10}`, domain];

	let meta: Record<string, unknown> = {};
	try { meta = JSON.parse(row.metadata_json || "{}"); } catch {}

	const fm = [
		"---",
		`id: "${row.id || ""}"`,
		`source: "${(row.source || "").replace(/"/g, '\\"')}"`,
		`importance: ${row.importance || 0.5}`,
		`timestamp: "${fmtDate(row.timestamp || row.created_at)}"`,
		`scope: "${row.scope || "global"}"`,
		`domain: "${domain}"`,
		`keywords: [${keywords.map(k => `"${k}"`).join(", ")}]`,
		`tags: [${tags.map(t => `"${t}"`).join(", ")}]`,
		...(meta.migrated ? [`migrated: "${String(meta.migrated)}"`] : []),
		...(isTranscript(row.content || "") ? [`summarized: true`] : []),
		"---",
		"",
		content,
		"",
		domain !== "unclassified" ? `\n---\n> 领域: [[_moc/${domain}|${domain}]]` : "",
		"",
	].join("\n");

	if (fs.existsSync(fpath) && fs.readFileSync(fpath, "utf8") === fm) return domain;

	fs.writeFileSync(fpath, fm, "utf8");
	return domain;
}

// ─── 导出: 增量（只导出 last_id 之后的新记忆）───

/** 清理「同一记忆 id 存在多个文件名」的重复笔记；只删同一 id 的多余文件，不按缺失删除 */
function reconcileVault(): number {
	const rows = sqliteQuery<{ id: string; timestamp: string | null; created_at: string | null }>(
		"SELECT id, timestamp, created_at FROM working_memory WHERE superseded_by IS NULL",
	);
	const canonical = new Map<string, string>();
	for (const r of rows) canonical.set(String(r.id).slice(0, 8), memoryFileName(String(r.id), stampOf(r)));
	if (!fs.existsSync(BRAIN_MEMORIES)) return 0;

	let removed = 0;
	for (const dir of fs.readdirSync(BRAIN_MEMORIES)) {
		const full = path.join(BRAIN_MEMORIES, dir);
		if (!fs.statSync(full).isDirectory()) continue;
		for (const f of fs.readdirSync(full)) {
			const m = f.match(/^(.*)_([0-9a-f]{8})\.md$/);
			if (!m) continue;
			const want = canonical.get(m[2]);
			if (want !== undefined && want !== f) {
				fs.unlinkSync(path.join(full, f));
				removed++;
			}
		}
	}
	if (removed > 0) console.log(`[ai-memory-sync] 清理同 id 重复笔记 ${removed} 个`);
	return removed;
}

function exportNewMemories(): number {
	ensureDirs();
	fs.mkdirSync(MOC_DIR, { recursive: true });
	const touchedDomains = new Set<string>();

	if (!lastExportedId) {
		const rows = sqliteQuery<MemoryRow>(
			"SELECT id, content, importance, source, scope, timestamp, created_at, metadata_json FROM working_memory WHERE superseded_by IS NULL ORDER BY rowid ASC",
		);
		for (const row of rows) {
			const domain = exportMemoryToMarkdown(row);
			if (domain !== "unclassified") touchedDomains.add(domain);
			lastExportedId = row.id;
		}
		saveCursor();
		console.log(`[ai-memory-sync] 全量导出 ${rows.length} 条记忆`);
		reconcileVault();
		for (const d of touchedDomains) updateMocForDomain(d);
		updateMocIndex();
		gitCommit(rows.length > 0 ? `sync: full export ${rows.length} memories` : "sync: vault reconcile");
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
		`SELECT id, content, importance, source, scope, timestamp, created_at, metadata_json FROM working_memory WHERE superseded_by IS NULL AND rowid > ? ORDER BY rowid ASC`,
		[cursorRowid],
	);
	for (const row of rows) {
		const domain = exportMemoryToMarkdown(row);
		if (domain !== "unclassified") touchedDomains.add(domain);
		lastExportedId = row.id;
		if (domain === "unclassified") domainStore.unclassifiedSinceCluster++;
	}
	if (rows.length > 0) console.log(`[ai-memory-sync] 增量导出 ${rows.length} 条记忆`);
	saveCursor();
	reconcileVault();
	for (const d of touchedDomains) updateMocForDomain(d);
	updateMocIndex();
	gitCommit(rows.length > 0 ? `sync: ${rows.length} new memories` : "sync: vault reconcile");
	const stats = getStats();
	if (shouldCluster(stats.total)) {
		saveDomainStore(domainStore);
		void discoverDomains().catch(e => console.error(`[ai-memory-sync] 聚类失败: ${e instanceof Error ? e.message : String(e)}`));
	}
	return rows.length;
}

// ─── 导入: inbox/ → Mnemopi（带分块 + 去重） ───

async function importInbox(): Promise<number> {
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
		const chunks = semanticChunks(content, CFG.chunkThreshold);
		let chunkErrors = 0;
		for (const chunk of chunks) {
			try {
				const id = await dedupRemember(chunk, source, importance);
				console.log(`[ai-memory-sync] dedupRemember() OK: ${id} (source=${source})`);
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

// ─── 防抖导出 ───

/** 防抖导出：用会话托管定时器，回调抛错由宿主收口，不会带崩整个会话 */
function scheduleExport(ctx: ExtensionContext): void {
	if (exportTimer) ctx.clearTimer(exportTimer);
	exportTimer = ctx.setTimeout(() => {
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

// ─── [新] 数据清洗：对现有记忆做去重/摘要/分块 ───

/** 清洗现有数据：transcript 摘要 + importance 重算 */
function cleanExistingMemories(): { summarized: number; deduped: number; decayed: number } {
	const result = { summarized: 0, deduped: 0, decayed: 0 };

	// 1. transcript 摘要：找到所有 transcript 记忆，提取摘要后更新
	const transcripts = sqliteQuery<{ id: string; content: string }>(
		"SELECT id, content FROM working_memory WHERE superseded_by IS NULL AND content LIKE '%[role: user]%'",
	);
	for (const t of transcripts) {
		const summary = summarizeTranscript(t.content);
		if (summary !== null && summary !== t.content) {
			mnemopi().update(t.id, summary, null);
			result.summarized++;
		} else if (summary === null) {
			// 无价值对话：标记为 superseded（软删除）
			withWriteDb(handle => handle.run("UPDATE working_memory SET superseded_by = 'cleanup-noise' WHERE id = ?", [t.id]));
			result.summarized++;
		}
	}

	// 2. importance 重算：根据 recall_count 重新分配
	const allRows = sqliteQuery<{ id: string; recall_count: number; importance: number; timestamp: string }>(
		"SELECT id, recall_count, importance, timestamp FROM working_memory WHERE superseded_by IS NULL",
	);
	for (const r of allRows) {
		const rc = r.recall_count ?? 0;
		let newImp = r.importance;
		if (rc > 3) newImp = Math.min(1.0, r.importance + 0.05 * Math.min(rc - 3, 5));
		else if (rc === 0) {
			const ageH = (Date.now() - new Date(r.timestamp).getTime()) / 3600000;
			if (ageH > 24) newImp = Math.max(0.1, r.importance - 0.05);
		}
		if (newImp !== r.importance) {
			mnemopi().update(r.id, null, newImp);
			result.decayed++;
		}
	}

	console.log(`[ai-memory-sync] 数据清洗完成: 摘要 ${result.summarized} 条, importance 调整 ${result.decayed} 条`);
	return result;
}

// ─── 扩展入口 ───

export default function (pi: ExtensionAPI): void {
	ensureDirs();

	// 初始化: 增量导出（游标为空时才全量）
	try {
		const count = exportNewMemories();
		initDone = true;
		console.log(`[ai-memory-sync] 初始化完成，导出 ${count} 条记忆到 ${CFG.vaultDir}`);
	} catch (e) {
		const msg = e instanceof Error ? e.message : String(e);
		console.error(`[ai-memory-sync] 初始化失败: ${msg}`);
	}

	// 钩子 1: retain / learn 工具完成后 → 防抖导出
	pi.on("tool_result", (event: Record<string, unknown>, ctx: ExtensionContext) => {
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

		scheduleExport(ctx);
	});

	// 钩子 2: 回合开始前 → 主动检索并把相关记忆注入本回合上下文
	pi.on("before_agent_start", async (event, _ctx) => {
		if (!CFG.proactiveRecall) return;
		const prompt = typeof event.prompt === "string" ? event.prompt.trim() : "";
		if (!prompt) return;
		const results = await vectorRecall(prompt.slice(0, 500), CFG.proactiveTopK);
		if (results.length === 0) return;
		const lines = results.map((r, i) => {
			const score = r.score ? r.score.toFixed(2) : "?";
			const content = (r.content ?? "").slice(0, 200);
			return `${i + 1}. [${score}] ${content}${r.truncated ? "…" : ""}`;
		});
		console.log(`[ai-memory-sync] 主动检索注入 ${results.length} 条记忆`);
		return {
			message: {
				customType: "ai-memory-sync.recall",
				content: `[相关记忆 ${results.length} 条]\n${lines.join("\n")}`,
				display: false,
			},
		};
	});

	// 钩子 3: 每回合结束 → 检查 inbox 导入 + importance 衰减
	pi.on("turn_end", (_event: unknown, _ctx: unknown) => {
		try {
			// importance 衰减
			decayImportance();
			// inbox 导入
			if (fs.existsSync(INBOX)) {
				const files = fs.readdirSync(INBOX);
				if (files.some(f => f.endsWith(".md"))) {
					void importInbox().catch(e => console.error(`[ai-memory-sync] inbox 导入失败: ${e instanceof Error ? e.message : String(e)}`));
				}
			}
		} catch (e) {
			const msg = e instanceof Error ? e.message : String(e);
			console.error(`[ai-memory-sync] turn_end 处理失败: ${msg}`);
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
		handler: async (_args, ctx) => {
			const imported = await importInbox();
			const exported = exportNewMemories();
			const stats = getStats();
			const msg = `同步完成: 导入 ${imported} 条, 导出 ${exported} 条, 总计 ${stats.total} 条记忆`;
			console.log(`[ai-memory-sync] ${msg}`);
			ctx.ui?.notify(msg, "info");
		},
	});

	// 命令: /memory-export — 仅导出
	pi.registerCommand("memory-export", {
		description: "手动导出 Mnemopi 记忆到 Obsidian vault（单向: Mnemopi → Markdown）",
		handler: async (_args, ctx) => {
			const exported = exportNewMemories();
			const msg = `导出完成: ${exported} 条记忆`;
			console.log(`[ai-memory-sync] ${msg}`);
			ctx.ui?.notify(msg, "info");
		},
	});

	// 命令: /memory-import — 仅导入
	pi.registerCommand("memory-import", {
		description: "手动导入 Obsidian inbox 中的笔记到 Mnemopi（单向: Markdown → Mnemopi）",
		handler: async (_args, ctx) => {
			const imported = await importInbox();
			const msg = `导入完成: ${imported} 条`;
			console.log(`[ai-memory-sync] ${msg}`);
			ctx.ui?.notify(msg, "info");
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
				ctx.ui?.notify("用法: /memory-recall <查询文本> [结果数量]", "info");
				return;
			}
			const results = await vectorRecall(query, topK);
			if (results.length === 0) {
				ctx.ui?.notify("未找到匹配的记忆", "info");
				return;
			}
			const lines = results.map((r, i) => {
				const score = r.score ? r.score.toFixed(3) : "?";
				const content = (r.content ?? "").slice(0, 150);
				return `${i + 1}. [${score}] ${content}${r.truncated ? "…" : ""}`;
			});
			const msg = lines.join("\n");
			console.log(`[ai-memory-sync] 检索 "${query}" → ${results.length} 条`);
			ctx.ui?.notify(msg, "info");
		},
	});

	// 命令: /memory-config — 显示配置
	pi.registerCommand("memory-config", {
		description: "显示 AI Memory Sync 扩展的当前配置和统计信息",
		handler: async (_args, ctx) => {
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
				`  去重阈值: ${CFG.dedupThreshold}`,
				`  衰减启用: ${CFG.decayEnabled ? "开" : "关"}`,
				`  分块阈值: ${CFG.chunkThreshold} 字符`,
				"",
				"统计:",
				`  Mnemopi 总记忆: ${stats.total} 条`,
				`  最新记忆时间: ${stats.lastExport || "无"}`,
				`  初始化状态: ${initDone ? "完成" : "未完成"}`,
				`  领域聚类阈值: ${CFG.domainClusterThreshold} 条`,
				`  已发现领域: ${domainStore.domains.length} 个`,
			];
			const msg = lines.join("\n");
			console.log(`[ai-memory-sync]\n${msg}`);
			ctx.ui?.notify(msg, "info");
		},
	});

	// 命令: /memory-domains — 查看领域分布统计
	pi.registerCommand("memory-domains", {
		description: "查看知识领域分布统计和 AI 发现的领域",
		handler: async (_args, ctx) => {
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
					.map(([name, count]) => `  ${name}: ${count} 条 (${total > 0 ? Math.round(count / total * 100) : 0}%)`),
			];
			if (domainStore.domains.length > 0) {
				lines.push("", "AI 发现的领域关键词:");
				for (const d of domainStore.domains) {
					lines.push(`  ${d.name} (${d.memberCount} 条): ${d.keywords.slice(0, 5).join(", ")}`);
				}
			}
			const msg = lines.join("\n");
			console.log(`[ai-memory-sync]\n${msg}`);
			ctx.ui?.notify(msg, "info");
		},
	});

	// 命令: /memory-cluster — 手动触发领域聚类
	pi.registerCommand("memory-cluster", {
		description: "手动触发 AI 领域聚类（用嵌入向量对所有记忆做 K-means 聚类）",
		handler: async (_args, ctx) => {
			ctx.ui?.notify("开始领域聚类，请稍候...", "info");
			try {
				await discoverDomains();
				const msg = domainStore.domains.length > 0
					? `聚类完成: 发现 ${domainStore.domains.length} 个领域`
					: "聚类完成: 记忆数量不足或无法生成领域";
				console.log(`[ai-memory-sync] ${msg}`);
				ctx.ui?.notify(msg, "info");
			} catch (e: unknown) {
				const msg = e instanceof Error ? e.message : String(e);
				console.error(`[ai-memory-sync] 聚类失败: ${msg}`);
				ctx.ui?.notify(`聚类失败: ${msg}`, "error");
			}
		},
	});

	// 命令: /memory-clean — 清洗现有数据（摘要 + 衰减 + 去重）
	pi.registerCommand("memory-clean", {
		description: "清洗现有记忆数据（transcript 摘要提取 + importance 重算 + 噪声标记）",
		handler: async (_args, ctx) => {
			ctx.ui?.notify("开始数据清洗...", "info");
			try {
				const result = cleanExistingMemories();
				// 重新导出清洗后的数据
				lastExportedId = null;
				const exported = exportNewMemories();
				const msg = `清洗完成: 摘要 ${result.summarized} 条, importance 调整 ${result.decayed} 条, 重新导出 ${exported} 条`;
				console.log(`[ai-memory-sync] ${msg}`);
				ctx.ui?.notify(msg, "info");
			} catch (e: unknown) {
				const msg = e instanceof Error ? e.message : String(e);
				console.error(`[ai-memory-sync] 清洗失败: ${msg}`);
				ctx.ui?.notify(`清洗失败: ${msg}`, "error");
			}
		},
	});
}
