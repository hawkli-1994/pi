import { mkdtemp, readdir, readFile, rm, writeFile } from "node:fs/promises";
import { createServer } from "node:http";
import type { AddressInfo } from "node:net";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { Type } from "typebox";
import { expect, test, vi } from "vitest";
import type { AgentSessionEvent } from "../src/core/agent-session.ts";
import { estimateProjectedContextTokens, estimateTokens } from "../src/core/compaction/compaction.ts";
import { ModelRuntime } from "../src/core/model-runtime.ts";
import { DefaultResourceLoader } from "../src/core/resource-loader.ts";
import { createAgentSession } from "../src/core/sdk.ts";
import { SessionManager } from "../src/core/session-manager.ts";
import { SettingsManager } from "../src/core/settings-manager.ts";
import { allowNetwork } from "./test-network-env.ts";

type Scenario = "threshold" | "length" | "repeated-length" | "summary-failure";

// Exercise the SDK's real HTTP provider and persistent projection, not a streamFn
// whose fake usage shares the compactor's character-counting assumption.
async function createPressureSession(scenario: Scenario, persisted = false) {
	allowNetwork();
	const cwd = await mkdtemp(join(tmpdir(), "pi-context-pressure-"));
	const settings = SettingsManager.inMemory({ retry: { enabled: false } });
	let completed = 0;
	let summaries = 0;
	let stepsAtSummary = 0;
	let truncations = 0;
	let failSummary = scenario === "summary-failure";
	const requests: { summary: boolean; maxTokens: number }[] = [];
	const serverErrors: unknown[] = [];
	const events: AgentSessionEvent[] = [];
	const upstream = createServer((req, res) => {
		void (async () => {
			const chunks: Buffer[] = [];
			for await (const chunk of req) chunks.push(Buffer.from(chunk));
			const body = JSON.parse(Buffer.concat(chunks).toString()) as {
				max_tokens?: number;
				max_completion_tokens?: number;
			};
			const summary = JSON.stringify(body).includes("context summarization assistant");
			requests.push({ summary, maxTokens: body.max_tokens ?? body.max_completion_tokens ?? 0 });
			if (summary && failSummary) {
				res.writeHead(403, { "content-type": "application/json" });
				res.end(JSON.stringify({ error: { message: "summary unavailable" } }));
				return;
			}
			if (summary) {
				summaries++;
				stepsAtSummary = completed;
			}
			const truncated =
				!summary &&
				scenario !== "threshold" &&
				(truncations === 0 || (scenario === "repeated-length" && truncations === 1));
			if (truncated) truncations++;
			const tool = !summary && !truncated && completed < 32;
			const finish = truncated ? "length" : tool ? "tool_calls" : "stop";
			const delta = tool
				? {
						role: "assistant",
						tool_calls: [
							{
								index: 0,
								id: `step-${completed + 1}`,
								type: "function",
								function: { name: "record_step", arguments: JSON.stringify({ step: completed + 1 }) },
							},
						],
					}
				: {
						role: "assistant",
						content: summary
							? `Verified 1-${completed}. Next ${completed + 1}. Goal 32.`
							: truncated
								? "."
								: "All 32 files verified.",
					};
			// Independent token counts deliberately disagree with chars/4. Include cached
			// prompt tokens: total input, not just uncached input, consumes the window.
			const input = summaries === 0 ? 63455 : summary ? 16000 : 3000 + (completed - stepsAtSummary) * 7000;
			const output = truncated ? 1 : 279;
			res.writeHead(200, { "content-type": "text/event-stream" });
			res.write(
				`data: ${JSON.stringify({ id: "pressure", model: "pressure", choices: [{ index: 0, delta, finish_reason: null }] })}\n\n`,
			);
			res.end(
				`data: ${JSON.stringify({
					id: "pressure",
					model: "pressure",
					choices: [{ index: 0, delta: {}, finish_reason: finish }],
					usage: {
						prompt_tokens: input,
						completion_tokens: output,
						total_tokens: input + output,
						prompt_tokens_details: { cached_tokens: Math.max(0, input - 4000) },
					},
				})}\n\ndata: [DONE]\n\n`,
			);
		})().catch((error) => {
			serverErrors.push(error);
			res.destroy();
		});
	});
	await new Promise<void>((resolve) => upstream.listen(0, "127.0.0.1", resolve));
	const model = {
		id: "pressure",
		name: "Pressure",
		provider: "pressure",
		api: "openai-completions" as const,
		baseUrl: `http://127.0.0.1:${(upstream.address() as AddressInfo).port}/v1`,
		reasoning: false,
		input: ["text" as const],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 65536,
		maxTokens: 4096,
	};
	const runtime = await ModelRuntime.create({
		authPath: join(cwd, "auth.json"),
		modelsPath: null,
		allowModelNetwork: false,
	});
	runtime.registerProvider(model.provider, { api: model.api, baseUrl: model.baseUrl, models: [model] });
	await runtime.setRuntimeApiKey(model.provider, "offline");
	const loader = new DefaultResourceLoader({
		cwd,
		agentDir: cwd,
		settingsManager: settings,
		noSkills: true,
		noContextFiles: true,
		noExtensions: true,
		noPromptTemplates: true,
		noThemes: true,
	});
	await loader.reload();
	let manager = persisted ? SessionManager.create(cwd, join(cwd, "sessions")) : SessionManager.inMemory(cwd);
	for (let i = 0; i < 18; i++) {
		manager.appendMessage({
			role: "user",
			content: "数据库事务日志索引调度一致性故障恢复".repeat(160),
			timestamp: Date.now(),
		});
		manager.appendMessage({
			role: "assistant",
			content: [{ type: "text", text: "已记录。" }],
			timestamp: Date.now(),
			api: model.api,
			provider: model.provider,
			model: model.id,
			stopReason: "stop",
			usage: {
				input: 0,
				output: 0,
				cacheRead: 0,
				cacheWrite: 0,
				totalTokens: 0,
				cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
			},
		});
	}
	if (persisted) manager = SessionManager.open(manager.getSessionFile()!);
	const { session } = await createAgentSession({
		cwd,
		model,
		modelRuntime: runtime,
		resourceLoader: loader,
		settingsManager: settings,
		sessionManager: manager,
		tools: ["record_step"],
		customTools: [
			{
				name: "record_step",
				label: "Record step",
				description: "Record and verify the next step.",
				parameters: Type.Object({ step: Type.Number() }),
				execute: async (_id, input) => {
					// The tool boundary validates this value against the declared schema.
					const { step } = input as { step: number };
					expect(step).toBe(completed + 1);
					await writeFile(join(cwd, `step-${step}.txt`), `verified ${step}`);
					completed = step;
					return {
						content: [{ type: "text" as const, text: `Verified ${completed}. ${"事务一致性".repeat(150)}` }],
						details: {},
					};
				},
			},
		],
	});
	session.subscribe((event) => events.push(event));
	return {
		session,
		manager,
		requests,
		events,
		serverErrors,
		cwd,
		get completed() {
			return completed;
		},
		get summaries() {
			return summaries;
		},
		allowSummary: () => {
			failSummary = false;
		},
		cleanup: async () => {
			await session.abort();
			session.dispose();
			upstream.closeAllConnections();
			await new Promise<void>((resolve) => upstream.close(() => resolve()));
			await rm(cwd, { recursive: true, force: true });
		},
	};
}

test.each([false, true])(
	"real provider transport completes 32 steps across usage-driven compaction (persisted=%s)",
	async (persisted) => {
		// Same-ms entries must be ordered by persisted provenance, not wall-clock guesses.
		vi.useFakeTimers({ toFake: ["Date"] });
		const run = await createPressureSession("threshold", persisted);
		try {
			await run.session.prompt("Create and verify all 32 files.");
			expect(
				run.completed,
				JSON.stringify({
					requests: run.requests,
					events: run.events.filter((event) => event.type === "compaction_end"),
					last: run.session.messages.at(-1),
				}),
			).toBe(32);
			expect(run.summaries).toBeGreaterThan(2);
			expect(run.requests.every((request) => request.maxTokens > 1)).toBe(true);
			expect(run.serverErrors).toEqual([]);
			expect(run.events.filter((event) => event.type === "compaction_end" && event.errorMessage)).toEqual([]);
			expect((await readdir(run.cwd)).filter((name) => /^step-\d+\.txt$/.test(name))).toHaveLength(32);
			for (let step = 1; step <= 32; step++)
				expect(await readFile(join(run.cwd, `step-${step}.txt`), "utf8")).toBe(`verified ${step}`);
		} finally {
			await run.cleanup();
			vi.useRealTimers();
		}
	},
);

test("length recovery retains pre-edit usage and continues the unfinished task", async () => {
	const run = await createPressureSession("length", true);
	try {
		await run.session.prompt("Create and verify all 32 files.");
		expect(run.completed).toBe(32);
		expect(
			run.events.some(
				(event) =>
					event.type === "compaction_end" && event.reason === "overflow" && event.result?.tokensBefore === 63456,
			),
		).toBe(true);
		expect(run.requests.every((request) => request.maxTokens > 1)).toBe(true);
	} finally {
		await run.cleanup();
	}
});

test("a failed summary leaves the truncated response projected and permits later recovery", async () => {
	const run = await createPressureSession("summary-failure", true);
	try {
		await run.session.prompt("Create and verify all 32 files.");
		expect(run.completed).toBe(0);
		expect(
			run.manager.getBranch().filter((entry) => entry.type === "context_edit" || entry.type === "compaction"),
		).toEqual([]);
		const projection = run.manager.buildSessionProjection();
		expect(estimateProjectedContextTokens(projection, run.manager.getBranch()).tokens).toBe(63456);
		expect(projection.messages.reduce((sum, message) => sum + estimateTokens(message), 0)).toBeLessThan(20000);
		expect(
			run.events.some(
				(event) => event.type === "compaction_end" && event.errorMessage?.includes("summary unavailable"),
			),
		).toBe(true);
		run.allowSummary();
		await run.session.prompt("Continue and verify all 32 files.");
		expect(run.completed).toBe(32);
	} finally {
		await run.cleanup();
	}
});

test("consecutive length stops exhaust one native retry without hiding the final failure", async () => {
	const run = await createPressureSession("repeated-length");
	try {
		await run.session.prompt("Create and verify all 32 files.");
		expect(run.completed).toBe(0);
		expect(run.events.filter((event) => event.type === "compaction_end" && event.result)).toHaveLength(1);
		expect(
			run.events.some(
				(event) =>
					event.type === "compaction_end" && event.errorMessage?.includes("after one compact-and-retry attempt"),
			),
		).toBe(true);
		expect(run.session.messages.at(-1)).toMatchObject({ role: "assistant", stopReason: "length" });
	} finally {
		await run.cleanup();
	}
});

test("cancelling recovery leaves the transcript and usage evidence intact", async () => {
	const run = await createPressureSession("length", true);
	const unsubscribe = run.session.subscribe((event) => {
		if (event.type === "compaction_start") run.session.abortCompaction();
	});
	try {
		await run.session.prompt("Create and verify all 32 files.");
		expect(run.completed).toBe(0);
		expect(
			run.manager.getBranch().filter((entry) => entry.type === "context_edit" || entry.type === "compaction"),
		).toEqual([]);
		expect(estimateProjectedContextTokens(run.manager.buildSessionProjection(), run.manager.getBranch()).tokens).toBe(
			63456,
		);
		expect(run.events.some((event) => event.type === "compaction_end" && event.aborted)).toBe(true);
		unsubscribe();
		await run.session.prompt("Continue and verify all 32 files.");
		expect(run.completed).toBe(32);
	} finally {
		unsubscribe();
		await run.cleanup();
	}
});
