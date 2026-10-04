import type {
	Api,
	AssistantMessage,
	AssistantMessageEvent,
	AssistantMessageEventStream,
	ImageContent,
	JsonValue,
	Message,
	Model,
	SimpleStreamOptions,
	TextContent,
	Tool,
	ToolResultMessage,
	TranscriptContext,
	Usage,
} from "@earendil-works/pi-ai";
import type { Static, TSchema } from "typebox";

/**
 * EN: Model boundary used by the loop. `Models.streamSimple` satisfies this contract. The normalized
 * transcript carries the system prompt and tool declarations in system messages, not separate context
 * fields.
 *
 * ZH: 循环调用模型的边界。Models.streamSimple 满足此契约。规范化对话记录通过 system 消息携带提示词和工具声明，而不是使用上下文上的独立字段。
 *
 * EN: Return an AssistantMessageEventStream, directly or through a promise. Encode request, model, and
 * runtime failures as error/aborted protocol events with a final message and errorMessage; do not throw or
 * reject for those failures.
 *
 * ZH: 直接或通过 Promise 返回 AssistantMessageEventStream。请求、模型和运行时失败应编码成 error/aborted 协议事件，并带最终消息与
 * errorMessage；这些失败不应直接抛出或拒绝。
 */
export type StreamFn = (
	model: Model<Api>,
	context: TranscriptContext,
	options?: SimpleStreamOptions,
) => AssistantMessageEventStream | Promise<AssistantMessageEventStream>;

/**
 * EN: Batch scheduling policy. Sequential mode completes each call before the next. Parallel mode validates
 * calls in order, executes allowed calls together, reports completion immediately, and records result
 * messages in original call order.
 *
 * ZH: 工具批次的调度策略。串行模式完成一个调用后再开始下一个；并行模式按顺序验证、并发执行获准调用、即时报告完成，并按原始调用顺序记录结果消息。
 */
export type ToolExecutionMode = "sequential" | "parallel";

/**
 * EN: Number of input messages consumed at a queue drain point. all consumes the current queue;
 * one-at-a-time consumes its oldest message and retains the rest for later polls.
 *
 * ZH: 队列消费点一次投递的消息数量。all 消费当前全部队列；one-at-a-time 只取最早的一条，其余保留到后续轮询。
 */
export type QueueMode = "all" | "one-at-a-time";

/** A single tool call content block emitted by an assistant message. */
export type AgentToolCall = Extract<AssistantMessage["content"][number], { type: "toolCall" }>;

/**
 * EN: Optional preflight decision. block prevents execution and creates an error tool result using reason
 * or a default message. A blocked result may request termination, effective only when every finalized call
 * in the batch requests it.
 *
 * ZH: 可选的预检查决定。block 阻止执行，并使用 reason 或默认文本生成工具错误结果。被阻止的结果可以请求终止，但仅当整批所有已完成调用都请求终止时才生效。
 */
export interface BeforeToolCallResult {
	block?: boolean;
	reason?: string;
	/**
	 * Hint that the agent should stop after the current tool batch when this call is blocked.
	 * Early termination only happens when every finalized tool result in the batch sets this to true.
	 */
	terminate?: boolean;
}

/**
 * EN: Field-level override of an executed result. Provided content, details, isError, usage, and terminate
 * replace their fields; omitted fields retain existing values. No deep merge occurs.
 *
 * ZH: 对已执行结果进行字段级覆盖。提供的 content、details、isError、usage 和 terminate 替换对应字段；未提供的字段沿用原值，不进行深层合并。
 *
 * EN: Provided structuredContent replaces the old value. If content changes without structuredContent, drop
 * the old structured value because it may no longer describe the visible result.
 *
 * ZH: 提供 structuredContent 时会替换旧值。若只改变 content 而未提供 structuredContent，则删除旧结构化值，因为它可能已无法描述当前可见结果。
 */
export interface AfterToolCallResult {
	content?: (TextContent | ImageContent)[];
	details?: unknown;
	structuredContent?: JsonValue;
	isError?: boolean;
	/** Usage from the final tool execution itself, if available. Not used for main LLM context accounting. */
	usage?: Usage;
	/**
	 * Hint that the agent should stop after the current tool batch.
	 * Early termination only happens when every finalized tool result in the batch sets this to true.
	 */
	terminate?: boolean;
}

/** Context passed to `beforeToolCall`. */
export interface BeforeToolCallContext {
	/** The assistant message that requested the tool call. */
	assistantMessage: AssistantMessage;
	/** The raw tool call block from `assistantMessage.content`. */
	toolCall: AgentToolCall;
	/** Validated tool arguments for the target tool schema. */
	args: unknown;
	/** Current agent context at the time the tool call is prepared. */
	context: AgentContext;
}

/** Context passed to `afterToolCall`. */
export interface AfterToolCallContext {
	/** The assistant message that requested the tool call. */
	assistantMessage: AssistantMessage;
	/** The raw tool call block from `assistantMessage.content`. */
	toolCall: AgentToolCall;
	/** Validated tool arguments for the target tool schema. */
	args: unknown;
	/** The executed tool result before any `afterToolCall` overrides are applied. */
	result: AgentToolResult<any>;
	/** Whether the executed tool result is currently treated as an error. */
	isError: boolean;
	/** Current agent context at the time the tool call is finalized. */
	context: AgentContext;
}

/**
 * EN: Completed turn snapshot passed to scheduling hooks. context includes the assistant and tool results.
 * newMessages includes only this invocation's output and, for a new prompt run, its initial input.
 *
 * ZH: 传给调度 hook 的已完成轮次信息。context 包含 assistant 与工具结果；newMessages 仅包含这次循环调用产生的消息，新 prompt 运行还包括其初始输入。
 */
export interface AgentTurnContext {
	/** The assistant message that completed the turn. */
	message: AssistantMessage;
	/** Tool result messages emitted for the completed turn. */
	toolResults: ToolResultMessage[];
	/** Current agent context after the turn's assistant message and tool results have been appended. */
	context: AgentContext;
	/** Messages that this loop invocation will return if it exits at this point. Prompt runs include the initial prompt messages; continuation runs do not include pre-existing context messages. */
	newMessages: AgentMessage[];
}

/** Decision returned by {@link FinishTurn}. Returning undefined preserves normal scheduling. */
export type AgentTurnDecision = { action: "continue" } | { action: "end" };

/**
 * EN: Run after the assistant and all finalized tool results, before turn_end. Return end to stop
 * immediately or continue to ensure one more provider request. Existing tool, steering, or follow-up
 * scheduling can satisfy that request.
 *
 * ZH: 在 assistant 与所有工具结果完成后、turn_end 前运行。返回 end 立即结束，返回 continue 保证再发生一次 Provider 请求。现有工具、steering 或
 * follow-up 调度可以满足这次续跑，不会额外重复请求。
 *
 * EN: Returning undefined keeps normal scheduling. Error and aborted responses are hard exits even if this
 * hook asks to continue.
 *
 * ZH: 返回 undefined 保留正常调度。错误和取消响应始终退出，即使这个 hook 要求继续也是如此。
 */
export type FinishTurn = (
	turn: AgentTurnContext,
	signal?: AbortSignal,
) => AgentTurnDecision | void | Promise<AgentTurnDecision | undefined> | Promise<void>;

/** Replacement runtime state used by the agent loop before starting another provider request. */
export interface AgentLoopTurnUpdate {
	/** Context for the next provider request. */
	context?: AgentContext;
	/** Messages to append before the next provider request, with normal lifecycle events. */
	messages?: AgentMessage[];
	/** Model for the next provider request. */
	model?: Model<any>;
	/** Thinking level for the next provider request. */
	thinkingLevel?: ThinkingLevel;
}

/** Runtime state available immediately before a conversational provider request. */
export interface PrepareRequestContext {
	context: AgentContext;
	model: Model<any>;
	thinkingLevel: ThinkingLevel;
}

/** Replacement runtime state for the provider request being prepared. */
export type AgentRequestUpdate = Omit<AgentLoopTurnUpdate, "messages">;

/**
 * EN: Refresh context, model, or thinking settings immediately before every provider request, including the
 * first. Pending input has already been appended and emitted. The hook does not poll input queues.
 *
 * ZH: 在每一次 Provider 请求前刷新上下文、模型或思考设置，包括第一次请求。此时待处理输入已追加并发出事件。这个 hook 不会轮询输入队列。
 */
export type PrepareRequest = (
	request: PrepareRequestContext,
	signal?: AbortSignal,
) => AgentRequestUpdate | void | Promise<AgentRequestUpdate | undefined> | Promise<void>;

export interface PrepareNextTurnContext extends AgentTurnContext {}

/**
 * EN: Policies used by the loop, separate from its transcript and executable tools. Read the request hooks
 * first, then turn scheduling and tool hooks. Context conversion occurs only at the provider boundary.
 *
 * ZH: 循环使用的策略，与对话历史及可执行工具分开保存。建议先读请求 hook，再读轮次调度与工具 hook。上下文转换只发生在 Provider 边界。
 *
 * EN: Input and conversion callbacks must provide safe fallback values instead of throwing. Raw loop stream
 * adapters do not recover arbitrary callback rejection into a normal event sequence.
 *
 * ZH: 输入及转换回调应提供安全的回退值，而不是抛出异常。原始循环的流适配器不会把任意回调拒绝恢复成正常的事件序列。
 */
export interface AgentLoopConfig extends SimpleStreamOptions {
	model: Model<any>;

	/**
	 * Converts AgentMessage[] to LLM-compatible Message[] before each LLM call.
	 *
	 * Each AgentMessage must be converted to a SystemMessage, UserMessage, AssistantMessage, or ToolResultMessage
	 * that the LLM can understand. AgentMessages that cannot be converted (e.g., UI-only notifications,
	 * status messages) should be filtered out.
	 *
	 * Contract: must not throw or reject. Return a safe fallback value instead.
	 * Throwing interrupts the low-level agent loop without producing a normal event sequence.
	 *
	 * @example
	 * ```typescript
	 * convertToLlm: (messages) => messages.flatMap(m => {
	 *   if (m.role === "custom") {
	 *     // Convert custom message to user message
	 *     return [{ role: "user", content: m.content, timestamp: m.timestamp }];
	 *   }
	 *   if (m.role === "notification") {
	 *     // Filter out UI-only messages
	 *     return [];
	 *   }
	 *   // Pass through standard LLM messages
	 *   return [m];
	 * })
	 * ```
	 */
	convertToLlm: (messages: AgentMessage[]) => Message[] | Promise<Message[]>;

	/**
	 * Optional transform applied to the context before `convertToLlm`.
	 *
	 * Use this for operations that work at the AgentMessage level:
	 * - Context window management (pruning old messages)
	 * - Injecting context from external sources
	 *
	 * Contract: must not throw or reject. Return the original messages or another
	 * safe fallback value instead.
	 *
	 * @example
	 * ```typescript
	 * transformContext: async (messages) => {
	 *   if (estimateTokens(messages) > MAX_TOKENS) {
	 *     return pruneOldMessages(messages);
	 *   }
	 *   return messages;
	 * }
	 * ```
	 */
	transformContext?: (messages: AgentMessage[], signal?: AbortSignal) => Promise<AgentMessage[]>;

	/**
	 * Resolves an API key dynamically for each LLM call.
	 *
	 * Useful for short-lived OAuth tokens (e.g., GitHub Copilot) that may expire
	 * during long-running tool execution phases.
	 *
	 * Contract: must not throw or reject. Return undefined when no key is available.
	 */
	getApiKey?: (provider: string) => Promise<string | undefined> | string | undefined;

	/**
	 * Called after the assistant message and all tool-result messages have been emitted, immediately before `turn_end`.
	 * `{ action: "end" }` ends the run without polling queues or preparing another request.
	 * On a normal turn, `{ action: "continue" }` ensures one next provider request. Tool-result, steering, or
	 * follow-up scheduling can satisfy that request and adds no extra request; otherwise the loop continues once
	 * with the current context. Returning undefined preserves normal scheduling. Error and aborted responses remain
	 * hard exits.
	 */
	finishTurn?: FinishTurn;

	/**
	 * Called immediately before every conversational provider request, including the first.
	 * Pending messages have already been appended. The returned context, model, and thinking level
	 * replace the runtime values for this and later requests in the run. This hook does not poll queues.
	 */
	prepareRequest?: PrepareRequest;

	/**
	 * Called after `turn_end` when the loop will continue, immediately before the next turn starts.
	 * Return replacement context/model/thinking state or messages to append to affect that turn.
	 * Return undefined to keep using the current context/config.
	 */
	prepareNextTurn?: (
		context: PrepareNextTurnContext,
	) => AgentLoopTurnUpdate | undefined | Promise<AgentLoopTurnUpdate | undefined>;

	/**
	 * Returns steering messages to inject into the conversation mid-run.
	 *
	 * Called after the current assistant turn finishes executing its tool calls, unless `finishTurn` ends the run.
	 * If messages are returned, they are added to the context before the next LLM call.
	 * Tool calls from the current assistant message are not skipped.
	 *
	 * Use this for "steering" the agent while it's working.
	 *
	 * Contract: must not throw or reject. Return [] when no steering messages are available.
	 */
	getSteeringMessages?: () => Promise<AgentMessage[]>;

	/**
	 * Returns follow-up messages to process after the agent would otherwise stop.
	 *
	 * Called when the agent has no more tool calls and no steering messages.
	 * If messages are returned, they're added to the context and the agent
	 * continues with another turn.
	 *
	 * Use this for follow-up messages that should wait until the agent finishes.
	 *
	 * Contract: must not throw or reject. Return [] when no follow-up messages are available.
	 */
	getFollowUpMessages?: () => Promise<AgentMessage[]>;

	/**
	 * Tool execution mode.
	 * - "sequential": execute tool calls one by one
	 * - "parallel": preflight tool calls sequentially, then execute allowed tools concurrently;
	 *   emit `tool_execution_end` in tool completion order after each tool is finalized,
	 *   then emit tool-result message artifacts later in assistant source order
	 *
	 * Default: "parallel"
	 */
	toolExecution?: ToolExecutionMode;

	/**
	 * Called before a tool is executed, after arguments have been validated.
	 *
	 * Return `{ block: true }` to prevent execution. The loop emits an error tool result instead.
	 * A blocked result can also set `terminate: true` to participate in the batch early-termination rule.
	 * The hook receives the agent abort signal and is responsible for honoring it.
	 */
	beforeToolCall?: (context: BeforeToolCallContext, signal?: AbortSignal) => Promise<BeforeToolCallResult | undefined>;

	/**
	 * Called after a tool finishes executing, before `tool_execution_end` and tool-result message events are emitted.
	 *
	 * Return an `AfterToolCallResult` to override parts of the executed tool result:
	 * - `content` replaces the full content array
	 * - `details` replaces the full details payload
	 * - `isError` replaces the error flag
	 * - `usage` replaces the tool result usage
	 * - `terminate` replaces the early-termination hint
	 *
	 * Any omitted fields keep their original values. No deep merge is performed.
	 * The hook receives the agent abort signal and is responsible for honoring it.
	 */
	afterToolCall?: (context: AfterToolCallContext, signal?: AbortSignal) => Promise<AfterToolCallResult | undefined>;
}

/**
 * Thinking/reasoning level for models that support it.
 * Note: "xhigh" and "max" are only supported by selected model families. Use model
 * thinking-level metadata from @earendil-works/pi-ai to detect support for a concrete model.
 */
export type ThinkingLevel = "off" | "minimal" | "low" | "medium" | "high" | "xhigh" | "max";

/**
 * Extensible interface for custom app messages.
 * Apps can extend via declaration merging:
 *
 * @example
 * ```typescript
 * declare module "@mariozechner/agent" {
 *   interface CustomAgentMessages {
 *     artifact: ArtifactMessage;
 *     notification: NotificationMessage;
 *   }
 * }
 * ```
 */
export interface CustomAgentMessages {
	// Empty by default - apps extend via declaration merging
}

/**
 * EN: Union of standard model messages and application-defined messages. Declaration merging extends
 * CustomAgentMessages. convertToLlm must translate or filter custom roles before a provider sees them.
 *
 * ZH: 标准模型消息与应用自定义消息的联合类型。通过声明合并扩展 CustomAgentMessages；在 Provider 收到消息前，convertToLlm 必须转换或过滤自定义角色。
 */
export type AgentMessage = Message | CustomAgentMessages[keyof CustomAgentMessages];

/**
 * EN: Public view of conversation data and current execution. The transcript owns system prompt and tool
 * declarations; tools holds executable implementations. Assigned message/tool arrays are shallow-copied.
 *
 * ZH: 公开的会话数据与当前执行状态。对话历史持有 system 提示词和工具声明，tools 持有可执行实现。给消息或工具数组赋值时采用浅拷贝。
 *
 * EN: isStreaming, streamingMessage, pendingToolCalls, and errorMessage are runtime observations.
 * isStreaming remains true through awaited agent_end listeners, until final cleanup.
 *
 * ZH: isStreaming、streamingMessage、pendingToolCalls 和 errorMessage 用来观察运行时。isStreaming 在等待 agent_end
 * 监听器期间仍为 true，直到最终清理才结束。
 */
export interface AgentState {
	/**
	 * Current system prompt, replayed from the transcript's system messages.
	 *
	 * Read-only: to change the prompt, append a system message with `content` or `sections`.
	 * In `initialState`, this seeds the leading system message.
	 */
	readonly systemPrompt: string;
	/** Active model used for future turns. */
	model: Model<any>;
	/** Requested reasoning level for future turns. */
	thinkingLevel: ThinkingLevel;
	/**
	 * Executable tools. Assigning a new array copies the top-level array.
	 *
	 * Differences from the tools declared in the transcript are announced to the model
	 * with a system message before the next request.
	 */
	set tools(tools: AgentTool<any>[]);
	get tools(): AgentTool<any>[];
	/**
	 * Conversation transcript. Assigning a new array copies the top-level array.
	 *
	 * System messages in the transcript carry the prompt and tool declarations.
	 */
	set messages(messages: AgentMessage[]);
	get messages(): AgentMessage[];
	/**
	 * True while the agent is processing a prompt or continuation.
	 *
	 * This remains true until awaited `agent_end` listeners settle.
	 */
	readonly isStreaming: boolean;
	/** Partial assistant message for the current streamed response, if any. */
	readonly streamingMessage?: AgentMessage;
	/** Tool call ids currently executing. */
	readonly pendingToolCalls: ReadonlySet<string>;
	/** Error message from the most recent failed or aborted assistant turn, if any. */
	readonly errorMessage?: string;
}

/**
 * EN: A partial or final tool result. content goes to the model; details supports presentation;
 * structuredContent serves programmatic callers. isError reports a failure without throwing and preserves
 * the structured result.
 *
 * ZH: 工具的部分或最终结果。content 传给模型，details 用于展示，structuredContent 服务于程序调用者。isError 可在不抛异常的情况下表达失败，并保留结构化结果。
 *
 * EN: terminate is a runtime scheduling hint. The agent skips automatic tool continuation only when every
 * finalized result in the batch sets it. It does not override queued input or an explicit finishTurn
 * continuation.
 *
 * ZH: terminate 是运行时调度提示。只有批次中每个已完成结果都设置它时，Agent 才跳过自动工具续跑；它不会覆盖排队输入或 finishTurn 显式要求的续跑。
 */
export interface AgentToolResult<T = JsonValue | undefined> {
	/** Text or image content returned to the model. */
	content: (TextContent | ImageContent)[];
	/** Arbitrary structured details for logs or UI rendering. */
	details: T;
	/**
	 * Machine-readable result matching the tool's `outputSchema`, for programmatic callers. Not sent
	 * to the model; `content` remains the model-facing result.
	 */
	structuredContent?: JsonValue;
	/** Usage from the final tool execution itself, if available. Not used for main LLM context accounting. */
	usage?: Usage;
	/**
	 * Report a failure without throwing. The model sees `content` as an error result, like a thrown
	 * error, but `details` and `structuredContent` are kept for the UI and programmatic callers.
	 */
	isError?: boolean;
	/**
	 * Hint that the agent should stop after the current tool batch.
	 * Early termination only happens when every finalized tool result in the batch sets this to true.
	 */
	terminate?: boolean;
}

/** Final outcome of a tool call after hooks ran. */
export interface AgentToolCallOutcome {
	toolCall: AgentToolCall;
	result: AgentToolResult<any>;
	isError: boolean;
}

/**
 * Callback used by tools to stream partial execution updates.
 *
 * The callback is scoped to the current `execute()` invocation. Calls made after
 * the tool promise settles are ignored.
 */
export type AgentToolUpdateCallback<T = any> = (partialResult: AgentToolResult<T>) => void;

/**
 * EN: Executable tool contract layered over the model-visible schema. The loop resolves name, optionally
 * prepares raw arguments, validates the schema, and only then calls execute with the abort signal and
 * progress callback.
 *
 * ZH: 在模型可见 schema 之上增加的可执行工具契约。循环按名称查找工具，可选地准备原始参数，验证 schema 后，才把取消信号和进度回调传给 execute。
 *
 * EN: A tool must throw or set isError on failure; error text alone does not mark failure. Updates after
 * execute settles are ignored. A sequential override makes the entire assistant tool batch sequential.
 *
 * ZH: 工具失败时必须抛异常或设置 isError；仅返回错误文本不会被标记为失败。execute 完成后的更新会被忽略。工具的串行覆盖设置会让整条 assistant 消息的工具批次串行执行。
 */
export interface AgentTool<TParameters extends TSchema = TSchema, TDetails = any> extends Tool<TParameters> {
	/** Human-readable label for UI display. */
	label: string;
	/**
	 * Optional compatibility shim for raw tool-call arguments before schema validation.
	 * Must return an object that matches `TParameters`.
	 */
	prepareArguments?: (args: unknown) => Static<TParameters>;
	/**
	 * JSON Schema of `structuredContent` in successful results. Tools that declare it should always
	 * set `structuredContent`.
	 */
	outputSchema?: TSchema;
	/**
	 * Execute the tool call. Throw on failure, or return a result with `isError: true`; do not only
	 * describe the failure in `content`.
	 */
	execute: (
		toolCallId: string,
		params: Static<TParameters>,
		signal?: AbortSignal,
		onUpdate?: AgentToolUpdateCallback<TDetails>,
	) => Promise<AgentToolResult<TDetails>>;
	/** Recovery policy for an effect whose durable intent exists but whose outcome is unknown. */
	replay?: "never" | "safe";
	/**
	 * Per-tool execution mode override.
	 * - "sequential": this tool must execute one at a time with other tool calls.
	 * - "parallel": this tool can execute concurrently with other tool calls.
	 *
	 * If omitted, the default execution mode applies.
	 */
	executionMode?: ToolExecutionMode;
}

/**
 * EN: Input and working state of a low-level loop. messages is the transcript, including prompt and
 * declared tools in system messages; tools contains the functions the host can actually execute.
 *
 * ZH: 底层循环的输入与工作状态。messages 是对话记录，其中的 system 消息保存提示词及工具声明；tools 则包含宿主实际能够执行的函数。
 */
export interface AgentContext {
	/** Transcript visible to the model. */
	messages: AgentMessage[];
	/** Tools available for execution in this run. */
	tools?: AgentTool<any>[];
}

/**
 * EN: Discriminated union for run, turn, message, and tool lifecycles. One run can contain many turns; one
 * turn contains one assistant response and its tool results. Only assistant streaming emits message_update.
 *
 * ZH: 区分运行、轮次、消息和工具生命周期的联合类型。一次运行可含多轮，一轮包含一次 assistant 响应及其工具结果；只有 assistant 的流式过程会发出 message_update。
 *
 * EN: agent_end is the final loop event. It does not mean awaited subscribers or Agent cleanup have
 * finished; use prompt settlement or waitForIdle for that boundary.
 *
 * ZH: agent_end 是循环最后一个事件，并不代表被等待的订阅者或 Agent 清理已经完成；应通过 prompt 完成或 waitForIdle 判断该边界。
 */
export type AgentEvent =
	// Agent lifecycle
	| { type: "agent_start" }
	| { type: "agent_end"; messages: AgentMessage[] }
	// Turn lifecycle - a turn is one assistant response + any tool calls/results
	| { type: "turn_start" }
	| { type: "turn_end"; message: AgentMessage; toolResults: ToolResultMessage[] }
	// Message lifecycle - emitted for system, user, assistant, and toolResult messages
	| { type: "message_start"; message: AgentMessage }
	// Only emitted for assistant messages during streaming
	| { type: "message_update"; message: AgentMessage; assistantMessageEvent: AssistantMessageEvent }
	| { type: "message_end"; message: AgentMessage }
	// Tool execution lifecycle
	| { type: "tool_execution_start"; toolCallId: string; toolName: string; args: any }
	| { type: "tool_execution_update"; toolCallId: string; toolName: string; args: any; partialResult: any }
	| { type: "tool_execution_end"; toolCallId: string; toolName: string; result: any; isError: boolean };
