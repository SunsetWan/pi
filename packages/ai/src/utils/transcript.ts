import type { Context, Message, SystemMessage, Tool, ToolReference, TranscriptContext } from "../types.ts";
import { contentText, getSystemMessageText } from "./text.ts";

export type { TranscriptContext } from "../types.ts";

/**
 * EN: Create a baseline system message for a nonempty prompt or tool set. Return undefined when both are
 * empty so normalization does not turn an empty conversation into a system-only one.
 *
 * ZH: 为非空提示词或工具集合创建基线 system 消息。两者皆空时返回 undefined，避免规范化把空对话变成仅含 system 的对话。
 */
export function createInitialSystemMessage(
	systemPrompt: string | undefined,
	tools: Tool[] | undefined,
): SystemMessage | undefined {
	const hasSystemPrompt = systemPrompt !== undefined && systemPrompt.length > 0;
	const hasTools = tools !== undefined && tools.length > 0;
	if (!hasSystemPrompt && !hasTools) return undefined;
	return {
		role: "system",
		content: systemPrompt ?? "",
		...(hasTools ? { toolsAdded: tools } : {}),
		timestamp: 0,
	};
}

/**
 * EN: Fold convenience prompt/tool fields into the transcript and return its branded provider-facing type.
 * It prepends a baseline only when those fields supply content; otherwise the message array is reused.
 *
 * ZH: 把便捷的提示词及工具字段合入对话历史，返回带品牌标记的 Provider 侧类型。只有这些字段提供内容时才前置基线消息，否则复用原消息数组。
 */
export function normalizeContext(context: Context): TranscriptContext {
	const initialMessage = createInitialSystemMessage(context.systemPrompt, context.tools);
	const messages = initialMessage ? [initialMessage, ...context.messages] : context.messages;
	return { messages } as TranscriptContext;
}

/**
 * Any message list. The replay helpers only read entries whose role is `"system"`, so
 * agent transcripts that carry custom message roles can be passed without filtering.
 */
export type TranscriptMessages = readonly { role: string }[];

function isSystemMessage(message: { role: string }): message is SystemMessage {
	return message.role === "system";
}

/** Return the leading system message, if the transcript starts with one. */
export function getInitialSystemMessage(messages: TranscriptMessages): SystemMessage | undefined {
	const first = messages[0];
	return first && isSystemMessage(first) ? first : undefined;
}

/** Drop the leading system message for APIs that carry the prompt outside the message list. */
export function withoutInitialSystemMessage(messages: Message[]): Message[] {
	return getInitialSystemMessage(messages) ? messages.slice(1) : messages;
}

/**
 * EN: Replay system-message tool changes by name. Each message removes names before adding definitions, so
 * a redefinition can replace the old schema in one step.
 *
 * ZH: 按名称重放 system 消息中的工具变化。每条消息先移除再新增，因此一次重新声明可以替换旧 schema。
 */
export function getCurrentTools(messages: TranscriptMessages): Tool[] {
	const tools = new Map<string, Tool>();
	for (const message of messages) {
		if (!isSystemMessage(message)) continue;
		for (const tool of message.toolsRemoved ?? []) tools.delete(tool.name);
		for (const tool of message.toolsAdded ?? []) tools.set(tool.name, tool);
	}
	return [...tools.values()];
}

/**
 * EN: Replay every system message into current prompt state. Append instruction content, replace/remove
 * named sections, and resolve the final tool set. Custom non-system roles are ignored.
 *
 * ZH: 把每条 system 消息重放成当前提示词状态：追加指令内容，替换或删除命名段落，解析最终工具集合。自定义的非 system 角色会被忽略。
 */
export function getCurrentSystemMessage(messages: TranscriptMessages): SystemMessage | undefined {
	const content: string[] = [];
	const sections = new Map<string, string>();
	let timestamp: number | undefined;
	for (const message of messages) {
		if (!isSystemMessage(message)) continue;
		timestamp ??= message.timestamp;
		const text = contentText(message.content);
		if (text.length > 0) content.push(text);
		for (const [name, value] of Object.entries(message.sections ?? {})) {
			if (value === null) sections.delete(name);
			else sections.set(name, value);
		}
	}
	const tools = getCurrentTools(messages);
	if (timestamp === undefined && tools.length === 0) return undefined;
	return {
		role: "system",
		content: content.join("\n\n"),
		...(sections.size > 0 ? { sections: Object.fromEntries(sections) } : {}),
		...(tools.length > 0 ? { toolsAdded: tools } : {}),
		timestamp: timestamp ?? 0,
	};
}

/** Render the current system prompt text after replaying every system message. */
export function getCurrentSystemPrompt(messages: TranscriptMessages): string {
	const message = getCurrentSystemMessage(messages);
	return message ? getSystemMessageText(message) : "";
}

/**
 * Rebuild the transcript for APIs without mid-conversation system messages: the replayed
 * system message leads, and every later system message is dropped.
 */
export function collapseSystemMessages(context: TranscriptContext): TranscriptContext {
	const head = getCurrentSystemMessage(context.messages);
	const messages = context.messages.filter((message) => message.role !== "system");
	return { messages: head ? [head, ...messages] : messages } as TranscriptContext;
}

/**
 * EN: Choose the representation a model can accept. Keep system updates in their historical positions when
 * supported; otherwise rebuild one current leading system message and remove later system entries.
 *
 * ZH: 选择模型能够接收的表示。支持时保留 system 更新的历史位置，否则重建一条当前 system 首消息并移除后续 system 条目。
 */
export function resolveTranscript(
	context: TranscriptContext,
	supportsMidConvoSystemMessages: boolean | undefined,
): TranscriptContext {
	return supportsMidConvoSystemMessages ? context : collapseSystemMessages(context);
}

/**
 * EN: Project an executable tool onto its model-visible schema. Drop execution and display fields, and
 * JSON-normalize parameters before transcript comparison or persistence.
 *
 * ZH: 把可执行工具投影为模型可见 schema。去掉执行和展示字段，并在对话历史比较或持久化前用 JSON 规范化参数。
 */
export function toToolDeclaration(tool: Tool): Tool {
	return {
		name: tool.name,
		description: tool.description,
		parameters: JSON.parse(JSON.stringify(tool.parameters)) as Tool["parameters"],
		...(tool.constrainedSampling === undefined ? {} : { constrainedSampling: tool.constrainedSampling }),
	};
}

/**
 * Whether two tools declare the same interface to the model.
 *
 * Both sides go through {@link toToolDeclaration} first: its JSON round-trip drops the
 * typebox symbol keys and `undefined` fields that a structural comparison would see, and
 * builds both objects with the same key order, so comparing the serialized declarations
 * is exact. This avoids a deep-equal dependency in a browser-safe package.
 */
export function declarationsEqual(left: Tool, right: Tool): boolean {
	return JSON.stringify(toToolDeclaration(left)) === JSON.stringify(toToolDeclaration(right));
}

export interface ToolStateChanges {
	toolsAdded: Tool[];
	toolsRemoved: ToolReference[];
}

/**
 * EN: Compare complete old/new tool sets by name and normalized declaration. A changed definition appears
 * as both a removal and an addition so replay produces the new schema.
 *
 * ZH: 按名称和规范化声明比较完整的新旧工具集合。定义变化同时表现为移除和新增，使重放得到新的 schema。
 */
export function getToolStateChanges(previous: readonly Tool[], current: readonly Tool[]): ToolStateChanges {
	const previousTools = new Map(previous.map((tool) => [tool.name, tool]));
	const currentTools = new Map(current.map((tool) => [tool.name, tool]));
	return {
		toolsAdded: current
			.filter((tool) => {
				const previousTool = previousTools.get(tool.name);
				return previousTool === undefined || !declarationsEqual(previousTool, tool);
			})
			.map(toToolDeclaration),
		toolsRemoved: previous
			.filter((tool) => {
				const currentTool = currentTools.get(tool.name);
				return currentTool === undefined || !declarationsEqual(tool, currentTool);
			})
			.map((tool) => ({ name: tool.name })),
	};
}

/** Every definition referenced by transcript tool state, in first-declaration order. */
export function getDeclaredTools(messages: TranscriptMessages): Tool[] {
	const definitions = new Map<string, Tool>();
	for (const message of messages) {
		if (!isSystemMessage(message)) continue;
		for (const tool of message.toolsAdded ?? []) definitions.set(tool.name, tool);
	}
	return [...definitions.values()];
}

/**
 * Whether a tool name was declared twice with different definitions. A transport that can
 * only reference previously declared tools by name cannot replay such a history.
 *
 * @deprecated No built-in transport needs this anymore: Anthropic expresses redefinitions with
 * inline `tool_definition` blocks. Kept for API compatibility and will be removed in a future release.
 */
export function hasToolRedefinitions(messages: TranscriptMessages): boolean {
	const declared = new Map<string, Tool>();
	for (const message of messages) {
		if (!isSystemMessage(message)) continue;
		for (const tool of message.toolsAdded ?? []) {
			const previous = declared.get(tool.name);
			if (previous !== undefined && !declarationsEqual(previous, tool)) return true;
			declared.set(tool.name, tool);
		}
	}
	return false;
}

/** Whether tool history contains a removal or same-name redeclaration that an addition-only transport cannot replay. */
export function hasNonAdditiveToolChanges(messages: TranscriptMessages): boolean {
	const declared = new Set<string>();
	for (const message of messages) {
		if (!isSystemMessage(message)) continue;
		if ((message.toolsRemoved?.length ?? 0) > 0) return true;
		for (const tool of message.toolsAdded ?? []) {
			if (declared.has(tool.name)) return true;
			declared.add(tool.name);
		}
	}
	return false;
}

export interface TranscriptTools {
	/** Tools sent in the top-level request field. */
	requestTools: Tool[];
	/**
	 * Whether later system messages carry their own `toolsAdded` as in-place additions.
	 * When false, `requestTools` already holds the complete current tool set.
	 */
	anchorsAdditions: boolean;
}

/**
 * EN: Split initial request tools from later transcript additions only when the transport supports it and
 * history is addition-only. Any removal or repeated name falls back to the complete current tool set.
 *
 * ZH: 仅当协议支持且历史只有新增操作时，将初始请求工具与后续历史新增分开。出现移除或重复名称时，回退到完整的当前工具集合。
 */
export function resolveTranscriptTools(messages: TranscriptMessages, supportsToolAdditions: boolean): TranscriptTools {
	const anchorsAdditions = supportsToolAdditions && !hasNonAdditiveToolChanges(messages);
	return {
		requestTools: anchorsAdditions
			? (getInitialSystemMessage(messages)?.toolsAdded ?? [])
			: getCurrentTools(messages),
		anchorsAdditions,
	};
}
