/** Summarize oversized history in bounded requests without dropping source text. */
import { contentText } from "@earendil-works/pi-ai";
import { clampMaxTokensToContext } from "@earendil-works/pi-ai/api/simple-options";
import type { Api, AssistantMessage, Model, TranscriptContext, Usage } from "@earendil-works/pi-ai/compat";
import { isContextOverflow } from "@earendil-works/pi-ai/compat";
import { combineUsage } from "../usage-totals.ts";

/** Keep every source span, validate each intermediate result, and account for every summary call. */
export async function completeBoundedSummary(
	model: Model<Api>,
	source: string,
	buildContext: (text: string) => TranscriptContext,
	complete: (context: TranscriptContext) => Promise<AssistantMessage>,
	signal?: AbortSignal,
	label = "Summarization",
): Promise<AssistantMessage> {
	let usage: Usage | undefined;
	// Only successful summaries may be merged. In particular, a truncated checkpoint
	// must never replace the original transcript, even when it contains usable text.
	const summarize = async (text: string, part = false): Promise<AssistantMessage> => {
		signal?.throwIfAborted();
		const prefix = part
			? "This is a contiguous excerpt or ordered partial summaries. It may start or end inside a message. Summarize only the information present.\n\n"
			: "";
		const context = buildContext(`${prefix}${text}`);
		let available: number;
		try {
			available = clampMaxTokensToContext(model, buildContext(prefix), Number.MAX_SAFE_INTEGER);
		} catch {
			throw new Error("Summary instructions leave no room for source text in the model context");
		}
		let fits = true;
		try {
			// Reuse the transport's estimate and safety margin. Leave room for a useful
			// summary, but account for fixed instructions before dividing the remaining
			// space between source text and output on small windows.
			const minimumOutput = Math.min(model.maxTokens, 4096, Math.max(1, Math.floor(available / 2)));
			fits = clampMaxTokensToContext(model, context, model.maxTokens) >= minimumOutput;
		} catch {
			fits = false;
		}
		if (fits) {
			const response = await complete(context);
			usage = usage ? combineUsage(usage, response.usage) : response.usage;
			signal?.throwIfAborted();
			if (!isContextOverflow(response)) {
				if (response.content.some((block) => block.type === "toolCall")) {
					throw new Error(`${label} attempted to call a tool`);
				}
				if (response.stopReason !== "stop") {
					const reason =
						response.stopReason === "length"
							? "generation hit the token cap and the summary is incomplete"
							: (response.errorMessage ?? response.stopReason);
					throw new Error(`${label} failed: ${reason}`);
				}
				if (!contentText(response.content).trim()) throw new Error(`${label} returned an empty summary`);
				return response;
			}
			// A provider may count CJK or other content higher than our estimate.
			// Split this rejected input instead of retrying the same oversized request.
		}
		let middle = Math.floor(text.length / 2);
		// Do not split a UTF-16 surrogate pair, including when text is a single code point.
		const before = text.charCodeAt(middle - 1);
		const after = text.charCodeAt(middle);
		if (before >= 0xd800 && before <= 0xdbff && after >= 0xdc00 && after <= 0xdfff) middle--;
		if (middle === 0) throw new Error("Summary instructions leave no room for source text in the model context");
		const left = await summarize(text.slice(0, middle), true);
		const right = await summarize(text.slice(middle), true);
		const merged = `<earlier-summary>\n${contentText(left.content)}\n</earlier-summary>\n<later-summary>\n${contentText(right.content)}\n</later-summary>`;
		// A model that copies/expands its input must not cause an unbounded reduction loop.
		if (merged.length >= text.length) throw new Error("Summarization did not reduce the oversized history");
		return summarize(merged, true);
	};
	const result = await summarize(source);
	return { ...result, usage: usage ?? result.usage };
}
