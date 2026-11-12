import { contentText, normalizeContext } from "@earendil-works/pi-ai";
import { clampMaxTokensToContext } from "@earendil-works/pi-ai/api/simple-options";
import type { AssistantMessage, Model, TranscriptContext } from "@earendil-works/pi-ai/compat";
import { estimateContextTokens } from "@earendil-works/pi-ai/utils/estimate";
import { expect, test } from "vitest";
import { completeBoundedSummary } from "../src/core/compaction/bounded-summary.ts";

const model: Model<"openai-completions"> = {
	id: "offline",
	name: "Offline",
	provider: "offline",
	api: "openai-completions",
	baseUrl: "http://unused.invalid",
	reasoning: false,
	input: ["text"],
	contextWindow: 65536,
	maxTokens: 4096,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
};
const buildContext = (text: string): TranscriptContext =>
	normalizeContext({
		systemPrompt: "Summarize the excerpt. Preserve markers and order.",
		messages: [{ role: "user", content: text, timestamp: 1 }],
	});
const reply = (
	text: string,
	stopReason: AssistantMessage["stopReason"] = "stop",
	errorMessage?: string,
): AssistantMessage => ({
	role: "assistant",
	content: [{ type: "text", text }],
	stopReason,
	errorMessage,
	api: model.api,
	provider: model.provider,
	model: model.id,
	timestamp: 2,
	usage: {
		input: 100,
		output: 10,
		totalTokens: 110,
		cacheRead: 0,
		cacheWrite: 0,
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
	},
});
const source = Array.from(
	{ length: 64 },
	(_, index) => `MARK${String(index).padStart(3, "0")} ${"evidence ".repeat(3000)}\n`,
).join("");

test.each([
	[1000, 150],
	[4096, 615],
	[8192, 1229],
])("summarizes oversized history with feasible requests in a %s-token window", async (contextWindow, margin) => {
	const currentModel = { ...model, contextWindow, maxTokens: 4096 };
	let calls = 0;
	const result = await completeBoundedSummary(
		currentModel,
		"evidence ".repeat(12000),
		buildContext,
		async (context) => {
			calls++;
			const input = estimateContextTokens(context).tokens;
			const output = clampMaxTokensToContext(currentModel, context, currentModel.maxTokens);
			expect(input + margin + output).toBeLessThanOrEqual(contextWindow);
			expect(output).toBeGreaterThan(100);
			return reply("Checkpoint.");
		},
	);
	expect(calls).toBeGreaterThan(1);
	expect(contentText(result.content)).toBe("Checkpoint.");
});

test.each([65536, 262144])(
	"reduces an oversized history without losing ordered evidence at window %s",
	async (contextWindow) => {
		let calls = 0;
		const currentModel = { ...model, contextWindow };
		const result = await completeBoundedSummary(currentModel, source, buildContext, async (context) => {
			calls++;
			expect(clampMaxTokensToContext(currentModel, context, 4096)).toBe(4096);
			const markers = context.messages.flatMap((message) => contentText(message.content).match(/MARK\d{3}/g) ?? []);
			return reply(markers.join(" "));
		});
		expect(calls).toBeGreaterThan(1);
		expect(contentText(result.content)).toBe(
			Array.from({ length: 64 }, (_, index) => `MARK${String(index).padStart(3, "0")}`).join(" "),
		);
		expect(result.usage.totalTokens).toBe(calls * 110);
	},
);

test("splits again when the provider tokenizer rejects an apparently fitting request", async () => {
	let rejected = 0;
	const result = await completeBoundedSummary(model, source.slice(0, 120000), buildContext, async (context) => {
		if (estimateContextTokens(context).tokens > 10000) {
			rejected++;
			return reply("", "error", "maximum context length exceeded");
		}
		return reply("Checkpoint.");
	});
	expect(rejected).toBeGreaterThan(0);
	expect(contentText(result.content)).toBe("Checkpoint.");
});

test.each(["length", "error", "aborted"] as const)(
	"rejects an intermediate %s without reducing partial output",
	async (stopReason) => {
		let calls = 0;
		await expect(
			completeBoundedSummary(model, source, buildContext, async () => {
				calls++;
				return calls === 2 ? reply("Partial.", stopReason, "summary unavailable") : reply("Checkpoint.");
			}),
		).rejects.toThrow("Summarization failed");
		expect(calls).toBe(2);
	},
);

test("cancellation prevents further chunks and merging", async () => {
	const controller = new AbortController();
	let calls = 0;
	await expect(
		completeBoundedSummary(
			model,
			source,
			buildContext,
			async () => {
				calls++;
				controller.abort();
				return reply("Checkpoint.");
			},
			controller.signal,
		),
	).rejects.toThrow();
	expect(calls).toBe(1);
});

test("stops when summary expansion cannot reduce the input", async () => {
	await expect(
		completeBoundedSummary(model, source, buildContext, async (context) =>
			reply(context.messages.map((message) => contentText(message.content)).join("\n")),
		),
	).rejects.toThrow("did not reduce");
});

test("oversized instructions fail without sending unbounded requests", async () => {
	let calls = 0;
	await expect(
		completeBoundedSummary(
			model,
			"😀",
			(text) => buildContext(`${source}\n${text}`),
			async () => {
				calls++;
				return reply("Checkpoint.");
			},
		),
	).rejects.toThrow("no room");
	expect(calls).toBe(0);
});
