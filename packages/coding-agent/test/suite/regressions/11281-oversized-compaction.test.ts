import { contentText, fauxAssistantMessage, type TranscriptContext } from "@earendil-works/pi-ai";
import { estimateContextTokens } from "@earendil-works/pi-ai/utils/estimate";
import { expect, test } from "vitest";
import { createHarness } from "../harness.ts";

// #11281: the summary request itself can exceed the model window.
test.each([65536, 262144])(
	"compacts oversized history through bounded requests at %s tokens",
	async (contextWindow) => {
		const harness = await createHarness({
			models: [{ id: "faux-1", contextWindow, maxTokens: 4096 }],
			settings: { compaction: { enabled: true, reserveTokens: 16384, keepRecentTokens: 1 } },
		});
		try {
			const markers = Array.from({ length: 64 }, (_, index) => `MARK${String(index).padStart(3, "0")}`);
			const source = markers.map((marker) => `${marker} ${"evidence ".repeat(3000)}\n`).join("");
			harness.sessionManager.appendMessage({ role: "user", content: source, timestamp: 1 });
			harness.sessionManager.appendMessage(fauxAssistantMessage("Recorded history", { timestamp: 2 }));
			harness.sessionManager.appendMessage({ role: "user", content: "Continue from the checkpoint", timestamp: 3 });
			harness.session.refreshContext();
			const summarize = (context: TranscriptContext) => {
				if (estimateContextTokens(context).tokens + 4096 > contextWindow) {
					return fauxAssistantMessage("", {
						stopReason: "error",
						errorMessage: "maximum context length exceeded",
					});
				}
				const found = context.messages.flatMap((message) => contentText(message.content).match(/MARK\d{3}/g) ?? []);
				return fauxAssistantMessage(found.join(" ") || "Checkpoint.");
			};
			harness.setResponses(Array.from({ length: 128 }, () => summarize));

			const result = await harness.session.compact();

			expect(harness.faux.state.callCount).toBeGreaterThan(1);
			expect(result.summary.match(/MARK\d{3}/g)).toEqual(markers);
			expect(harness.sessionManager.getEntries().filter((entry) => entry.type === "compaction")).toHaveLength(1);
		} finally {
			harness.cleanup();
		}
	},
);
