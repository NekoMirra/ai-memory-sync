// AI Memory Sync 的 git 执行器：独立进程运行，omp 主进程只负责投递请求。
// 放在独立进程的原因：Windows 上在 Bun 事件循环里等待子进程会阻塞输入，
// 而 git 的 add/status/pull/push 在大仓库上要数秒。
//
// 与主进程的协作协议（改这里之前先确认两侧同时改）：
// 1. 主进程写 vault 文件一律「临时文件 + rename」，git 在任意时刻读到的都是完整文件，
//    所以这里不需要与写文件共享锁；
// 2. 请求队列 aims-queue.jsonl 由生产者追加，本进程用 rename 整份摘取，避免「读—删」之间丢行；
// 3. 冲突方向：rebase 期间的 -X theirs 指被重放的本地提交，即本地 vault 内容优先。
import { execFileSync } from "node:child_process";
import * as fs from "node:fs";
import * as path from "node:path";

const vault = process.argv[2];
if (!vault) {
	process.stderr.write("git-sync: missing vault path\n");
	process.exit(2);
}

// 紧急刹车：存在 .git/aims-disabled 时立刻退出，供排查卡顿时一键停掉全部 git 执行器。
// 顺带把队列清掉：禁用期间没有消费者，不清就会无限增长。
if (fs.existsSync(path.join(vault, ".git", "aims-disabled"))) {
	try { fs.unlinkSync(path.join(vault, ".git", "aims-queue.jsonl")); } catch {}
	process.stdout.write("[ai-memory-sync:git] disabled by .git/aims-disabled\n");
	process.exit(0);
}

const queueFile = path.join(vault, ".git", "aims-queue.jsonl");
const lockFile = path.join(vault, ".git", "aims-sync.lock");
const attempts = 4;
// 一轮里最多消费几批请求：运行期间新追加的请求不必等下一次同步
const drainRounds = 5;

function log(message) {
	process.stdout.write(`[ai-memory-sync:git] ${message}\n`);
}

function git(args, timeout = 120000) {
	try {
		return { ok: true, out: execFileSync("git", args, { cwd: vault, encoding: "utf8", timeout, maxBuffer: 32 * 1024 * 1024, stdio: ["ignore", "pipe", "pipe"] }) };
	} catch (error) {
		return { ok: false, out: "", err: String(error.stderr ?? error.message ?? error).trim() };
	}
}

function reason(result) {
	const line = (result.err ?? "").split("\n").find(l => /error|fatal|rejected|CONFLICT/i.test(l));
	return (line ?? (result.err ?? "").split("\n")[0] ?? "unknown").trim();
}

const STALE_LOCK_MS = 120000;

function sleepSync(ms) {
	Atomics.wait(new Int32Array(new SharedArrayBuffer(4)), 0, 0, ms);
}

/** 重试间隔：200ms 起翻倍，封顶 4s。毫秒级连打四次等于没重试 */
function backoff(attempt) {
	sleepSync(Math.min(200 * 2 ** (attempt - 1), 4000));
}

/**
 * 取锁。持有者进程已退出、或锁文件已过期（父进程被结束时 helper 会被连带杀掉，
 * 来不及在 finally 里 unlink），都直接把锁抢过来，避免队列长期卡死。
 */
function acquireLock() {
	const deadline = Date.now() + 180000;
	while (Date.now() < deadline) {
		try {
			fs.writeFileSync(lockFile, String(process.pid), { flag: "wx" });
			return true;
		} catch {
			if (isStale()) {
				try {
					fs.unlinkSync(lockFile);
					log("回收过期锁（持有者已退出）");
				} catch {
					// 抢锁失败说明别的进程刚建了新锁，等下一轮再看
					sleepSync(200);
				}
				continue;
			}
			sleepSync(200);
		}
	}
	return false;
}

/** 锁持有者已死，或锁文件超龄 */
function isStale() {
	let holder = 0;
	let age = 0;
	try {
		holder = Number.parseInt(fs.readFileSync(lockFile, "utf8"), 10) || 0;
		age = Date.now() - fs.statSync(lockFile).mtimeMs;
	} catch {
		return false;
	}
	if (age > STALE_LOCK_MS) return true;
	if (holder <= 0) return true;
	try {
		process.kill(holder, 0);
		return false;
	} catch {
		return true;
	}
}

/**
 * 原子摘取队列：rename 出的整份文件归本进程独占，生产者之后的追加写进新的队列文件。
 * rename 瞬间仍持有旧句柄的追加会落在摘取后的文件里，读完再取一次比对长度，把它接回队列。
 * 逐行解析，坏行回写而不是整批丢弃。
 */
function takeQueue() {
	const aside = `${queueFile}.${process.pid}.draining`;
	try {
		fs.renameSync(queueFile, aside);
	} catch {
		return [];
	}
	let text = "";
	try {
		text = fs.readFileSync(aside, "utf8");
		const again = fs.readFileSync(aside, "utf8");
		if (again.length > text.length) {
			try { fs.appendFileSync(queueFile, again.slice(text.length), "utf8"); } catch {}
		}
	} catch (e) {
		log(`读取队列失败: ${String(e.message ?? e)}`);
		return [];
	} finally {
		try { fs.unlinkSync(aside); } catch {}
	}

	const entries = [];
	const broken = [];
	for (const line of text.split("\n")) {
		if (line.trim() === "") continue;
		try {
			const entry = JSON.parse(line);
			if (entry !== null && typeof entry === "object") entries.push(entry);
			else broken.push(line);
		} catch {
			broken.push(line);
		}
	}
	if (broken.length > 0) {
		log(`队列有 ${broken.length} 行无法解析，已回写等待排查`);
		try { fs.appendFileSync(queueFile, `${broken.join("\n")}\n`, "utf8"); } catch {}
	}
	return entries;
}

/** 本地领先远端的提交数；远端引用不存在或 git 失败返回 -1 */
function localAhead(branch) {
	const ahead = git(["rev-list", "--count", `origin/${branch}..HEAD`], 20000);
	if (!ahead.ok) return -1;
	const count = Number.parseInt(ahead.out.trim(), 10);
	return Number.isFinite(count) ? count : -1;
}

function commitAndPush(message) {
	const head = git(["rev-parse", "--abbrev-ref", "HEAD"], 15000);
	const name = head.out.trim();
	const branch = head.ok && name !== "" && name !== "HEAD" ? name : "master";

	const add = git(["add", "-A"], 30000);
	if (!add.ok) return log(`add 失败: ${reason(add)}`);
	const status = git(["status", "--porcelain"], 20000);
	if (!status.ok) return log(`status 失败: ${reason(status)}`);

	if (status.out.trim() === "") {
		// 上一次推送失败的提交不会自己重来：工作区干净也要看本地是否仍领先远端
		const ahead = localAhead(branch);
		if (ahead === 0) return;
		if (ahead > 0) log(`工作区干净，补推上次未送达的 ${ahead} 个提交`);
	} else {
		const commit = git(["commit", "-m", message], 40000);
		if (!commit.ok) return log(`commit 失败: ${reason(commit)}`);
	}

	for (let attempt = 1; attempt <= attempts; attempt++) {
		git(["rebase", "--abort"], 15000);
		const pull = git(["pull", "--rebase", "--autostash", "-X", "theirs", "origin", branch], 180000);
		if (!pull.ok) {
			git(["rebase", "--abort"], 15000);
			const why = reason(pull);
			if (attempt < attempts) {
				log(`变基拉取失败（第 ${attempt}/${attempts} 次）: ${why}，退避后重试`);
				backoff(attempt);
				continue;
			}
			return log(`变基拉取连续失败，本地提交已保留: ${why}`);
		}
		const push = git(["push", "origin", branch], 180000);
		if (push.ok) return;
		const why = reason(push);
		if (attempt < attempts) {
			log(`推送失败（第 ${attempt}/${attempts} 次）: ${why}，退避后重试`);
			backoff(attempt);
			continue;
		}
		log(`推送连续失败，本地提交已保留，下次同步继续尝试: ${why}`);
	}
}

if (!acquireLock()) {
	log("等待锁超时，放弃本轮（下次同步会重试）");
	process.exit(0);
}

try {
	for (let round = 0; round < drainRounds; round++) {
		const entries = takeQueue();
		if (entries.length === 0) break;
		for (const entry of entries) commitAndPush(String(entry.message ?? "sync: vault update"));
	}
} finally {
	try { fs.unlinkSync(lockFile); } catch {}
}
