import {
	createInitialSystemMessage,
	getCurrentSystemMessage,
	getCurrentSystemPrompt,
	type ImageContent,
	type Message,
	type Model,
	type SimpleStreamOptions,
	type TextContent,
	type ThinkingBudgets,
	type Transport,
	toToolDeclaration,
} from "@earendil-works/pi-ai";
import { runAgentLoop, runAgentLoopContinue } from "./agent-loop.ts";
import { getDefaultStreamFn } from "./stream-fn.ts";
import type {
	AfterToolCallContext,
	AfterToolCallResult,
	AgentContext,
	AgentEvent,
	AgentLoopConfig,
	AgentLoopTurnUpdate,
	AgentMessage,
	AgentState,
	AgentTool,
	BeforeToolCallContext,
	BeforeToolCallResult,
	FinishTurn,
	PrepareNextTurnContext,
	PrepareRequest,
	QueueMode,
	StreamFn,
	ToolExecutionMode,
} from "./types.ts";

export type { QueueMode } from "./types.ts";

/**
 * EN: Keep the four model message roles and discard application-only messages before a provider request.
 * This filters by role; it does not validate message fields.
 *
 * ZH: 在请求 Provider 前保留模型使用的四种消息角色，并过滤应用自定义消息。这里只按角色筛选，不验证消息字段。
 */
function defaultConvertToLlm(messages: AgentMessage[]): Message[] {
	return messages.filter(
		(message) =>
			message.role === "system" ||
			message.role === "user" ||
			message.role === "assistant" ||
			message.role === "toolResult",
	);
}

const EMPTY_USAGE = {
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
};

const DEFAULT_MODEL = {
	id: "unknown",
	name: "unknown",
	api: "unknown",
	provider: "unknown",
	baseUrl: "",
	reasoning: false,
	input: [],
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
	contextWindow: 0,
	maxTokens: 0,
} satisfies Model<any>;

type MutableAgentState = Omit<AgentState, "isStreaming" | "streamingMessage" | "pendingToolCalls" | "errorMessage"> & {
	isStreaming: boolean;
	streamingMessage?: AgentMessage;
	pendingToolCalls: Set<string>;
	errorMessage?: string;
};

/**
 * EN: Initial values for {@link Agent}. A supplied transcript that already starts with a system message
 * owns its prompt and tool declarations. Otherwise initialization adds that baseline.
 *
 * ZH: 构造 Agent 时使用的初始值。若传入的历史已以 system 消息开头，则以这份历史中的提示词和工具声明为准；否则初始化会补入基线。
 */
export type AgentInitialState = Partial<
	Omit<AgentState, "pendingToolCalls" | "isStreaming" | "streamingMessage" | "errorMessage">
>;

/**
 * EN: Create mutable runtime state. Copy the message and tool arrays, but keep their objects shared. Replay
 * system messages when reading the prompt so it cannot drift from the transcript.
 *
 * ZH: 建立运行时可变状态。消息数组和工具数组采用浅拷贝，内部对象仍然共享。读取提示词时重放 system 消息，使提示词与对话记录保持一致。
 */
function createMutableAgentState(initialState?: AgentInitialState): MutableAgentState {
	let tools = initialState?.tools?.slice() ?? [];
	let messages = initialState?.messages?.slice() ?? [];
	const initialMessage = createInitialSystemMessage(initialState?.systemPrompt, tools.map(toToolDeclaration));
	if (messages[0]?.role !== "system" && initialMessage) messages.unshift(initialMessage);

	return {
		get systemPrompt() {
			return getCurrentSystemPrompt(messages);
		},
		model: initialState?.model ?? DEFAULT_MODEL,
		thinkingLevel: initialState?.thinkingLevel ?? "off",
		get tools() {
			return tools;
		},
		set tools(nextTools: AgentTool<any>[]) {
			tools = nextTools.slice();
		},
		get messages() {
			return messages;
		},
		set messages(nextMessages: AgentMessage[]) {
			messages = nextMessages.slice();
		},
		isStreaming: false,
		streamingMessage: undefined,
		pendingToolCalls: new Set<string>(),
		errorMessage: undefined,
	};
}

/**
 * EN: Dependencies and policies for {@link Agent}. `streamFn` supplies model responses; hooks control
 * context, tools, and turn scheduling. The agent owns lifecycle and state, not provider authentication or
 * session files.
 *
 * ZH: Agent 的依赖和运行策略。streamFn 提供模型响应；各个 hook 控制上下文、工具和轮次调度。Agent 管理生命周期与状态，Provider 认证和会话文件由外部负责。
 */
export interface AgentOptions {
	initialState?: AgentInitialState;
	convertToLlm?: (messages: AgentMessage[]) => Message[] | Promise<Message[]>;
	transformContext?: (messages: AgentMessage[], signal?: AbortSignal) => Promise<AgentMessage[]>;
	streamFn: StreamFn;
	getApiKey?: (provider: string) => Promise<string | undefined> | string | undefined;
	onPayload?: SimpleStreamOptions["onPayload"];
	onResponse?: SimpleStreamOptions["onResponse"];
	onProviderStreamEvent?: SimpleStreamOptions["onProviderStreamEvent"];
	beforeToolCall?: (context: BeforeToolCallContext, signal?: AbortSignal) => Promise<BeforeToolCallResult | undefined>;
	afterToolCall?: (context: AfterToolCallContext, signal?: AbortSignal) => Promise<AfterToolCallResult | undefined>;
	finishTurn?: FinishTurn;
	prepareRequest?: PrepareRequest;
	prepareNextTurn?: (
		signal?: AbortSignal,
	) => Promise<AgentLoopTurnUpdate | undefined> | AgentLoopTurnUpdate | undefined;
	prepareNextTurnWithContext?: (
		context: PrepareNextTurnContext,
		signal?: AbortSignal,
	) => Promise<AgentLoopTurnUpdate | undefined> | AgentLoopTurnUpdate | undefined;
	steeringMode?: QueueMode;
	followUpMode?: QueueMode;
	sessionId?: string;
	thinkingBudgets?: ThinkingBudgets;
	transport?: Transport;
	maxRetryDelayMs?: number;
	toolExecution?: ToolExecutionMode;
}

/**
 * EN: Store input until a loop scheduling point. `peek()` respects the delivery mode without removing
 * entries; `drain()` removes exactly that selection. Steering and follow-up use separate queues.
 *
 * ZH: 把输入保留到循环的调度点。peek() 按投递模式读取且不删除；drain() 仅移除本次选中的消息。steering 与 follow-up 各有独立队列。
 */
class PendingMessageQueue {
	private messages: AgentMessage[] = [];
	public mode: QueueMode;

	constructor(mode: QueueMode) {
		this.mode = mode;
	}

	enqueue(message: AgentMessage): void {
		this.messages.push(message);
	}

	hasItems(): boolean {
		return this.messages.length > 0;
	}

	peek(): AgentMessage[] {
		if (this.mode === "all") return this.messages.slice();
		const first = this.messages[0];
		return first ? [first] : [];
	}

	drain(): AgentMessage[] {
		const drained = this.peek();
		this.messages = this.messages.slice(drained.length);
		return drained;
	}

	clear(): void {
		this.messages = [];
	}
}

type ActiveRun = {
	promise: Promise<void>;
	resolve: () => void;
	abortController: AbortController;
};

/**
 * EN: Stateful owner of an agent run. Read `prompt()` first, then the private lifecycle methods and
 * `runAgentLoop()` in agent-loop.ts. Loop events update the public state before subscribers run.
 *
 * ZH: 一次 Agent 运行的状态所有者。先读 prompt()，再读私有生命周期方法和 agent-loop.ts 中的 runAgentLoop()。循环事件先更新公开状态，再通知订阅者。
 *
 * EN: Only one run is active at a time. Tools and queued input can cause several model turns in that run.
 * `agent_end` ends the event sequence; settlement also waits for its subscribers and runtime cleanup.
 *
 * ZH: 同一时刻只允许一次运行。工具和排队输入可能让这次运行包含多轮模型请求。agent_end 结束事件序列；真正完成还需等待该事件的订阅者和运行时清理。
 */
export class Agent {
	private _state: MutableAgentState;
	private readonly listeners = new Set<(event: AgentEvent, signal: AbortSignal) => Promise<void> | void>();
	private readonly steeringQueue: PendingMessageQueue;
	private readonly followUpQueue: PendingMessageQueue;

	public convertToLlm: (messages: AgentMessage[]) => Message[] | Promise<Message[]>;
	public transformContext?: (messages: AgentMessage[], signal?: AbortSignal) => Promise<AgentMessage[]>;
	public streamFunction: StreamFn;
	public getApiKey?: (provider: string) => Promise<string | undefined> | string | undefined;
	public onPayload?: SimpleStreamOptions["onPayload"];
	public onResponse?: SimpleStreamOptions["onResponse"];
	public onProviderStreamEvent?: SimpleStreamOptions["onProviderStreamEvent"];
	public beforeToolCall?: (
		context: BeforeToolCallContext,
		signal?: AbortSignal,
	) => Promise<BeforeToolCallResult | undefined>;
	public afterToolCall?: (
		context: AfterToolCallContext,
		signal?: AbortSignal,
	) => Promise<AfterToolCallResult | undefined>;
	public finishTurn?: FinishTurn;
	public prepareRequest?: PrepareRequest;
	public prepareNextTurn?: (
		signal?: AbortSignal,
	) => Promise<AgentLoopTurnUpdate | undefined> | AgentLoopTurnUpdate | undefined;
	public prepareNextTurnWithContext?: (
		context: PrepareNextTurnContext,
		signal?: AbortSignal,
	) => Promise<AgentLoopTurnUpdate | undefined> | AgentLoopTurnUpdate | undefined;
	private activeRun?: ActiveRun;
	/** Session identifier forwarded to providers for cache-aware backends. */
	public sessionId?: string;
	/** Optional per-level thinking token budgets forwarded to the stream function. */
	public thinkingBudgets?: ThinkingBudgets;
	/** Preferred transport forwarded to the stream function. */
	public transport: Transport;
	/** Optional cap for provider-requested retry delays. */
	public maxRetryDelayMs?: number;
	/** Tool execution strategy for assistant messages that contain multiple tool calls. */
	public toolExecution: ToolExecutionMode;

	constructor(options: AgentOptions) {
		// Older compiled consumers may omit options or streamFn even though the current API requires them.
		const runtimeOptions: Partial<AgentOptions> = options ?? {};
		this._state = createMutableAgentState(runtimeOptions.initialState);
		this.convertToLlm = runtimeOptions.convertToLlm ?? defaultConvertToLlm;
		this.transformContext = runtimeOptions.transformContext;
		this.streamFunction = runtimeOptions.streamFn ?? getDefaultStreamFn();
		this.getApiKey = runtimeOptions.getApiKey;
		this.onPayload = runtimeOptions.onPayload;
		this.onResponse = runtimeOptions.onResponse;
		this.onProviderStreamEvent = runtimeOptions.onProviderStreamEvent;
		this.beforeToolCall = runtimeOptions.beforeToolCall;
		this.afterToolCall = runtimeOptions.afterToolCall;
		this.finishTurn = runtimeOptions.finishTurn;
		this.prepareRequest = runtimeOptions.prepareRequest;
		this.prepareNextTurn = runtimeOptions.prepareNextTurn;
		this.prepareNextTurnWithContext = runtimeOptions.prepareNextTurnWithContext;
		this.steeringQueue = new PendingMessageQueue(runtimeOptions.steeringMode ?? "one-at-a-time");
		this.followUpQueue = new PendingMessageQueue(runtimeOptions.followUpMode ?? "one-at-a-time");
		this.sessionId = runtimeOptions.sessionId;
		this.thinkingBudgets = runtimeOptions.thinkingBudgets;
		this.transport = runtimeOptions.transport ?? "auto";
		this.maxRetryDelayMs = runtimeOptions.maxRetryDelayMs;
		this.toolExecution = runtimeOptions.toolExecution ?? "parallel";
	}

	/**
	 * EN: Observe events after internal state has been reduced. Listeners are awaited in insertion order and
	 * receive the active abort signal. The returned function removes this listener.
	 *
	 * ZH: 在内部状态更新后观察事件。监听器按注册顺序依次等待，并收到本次运行的取消信号。返回的函数用于取消订阅。
	 *
	 * EN: An asynchronous listener delays later loop phases and idle settlement, including after `agent_end`.
	 * Avoid waiting for this same agent to become idle from inside a listener.
	 *
	 * ZH: 异步监听器会推迟后续循环阶段和空闲状态，即使当前事件为 agent_end 也是如此。不要在监听器内部等待同一个 Agent 进入空闲。
	 */
	subscribe(listener: (event: AgentEvent, signal: AbortSignal) => Promise<void> | void): () => void {
		this.listeners.add(listener);
		return () => this.listeners.delete(listener);
	}

	/**
	 * EN: Current live state, not an immutable snapshot. Assigning messages or tools copies only the outer
	 * array; objects and arrays read from state remain mutable.
	 *
	 * ZH: 返回当前实时状态，而不是不可变快照。给 messages 或 tools 赋值时只复制外层数组；从状态中读出的数组及对象仍可修改。
	 */
	get state(): AgentState {
		return this._state;
	}

	/** Controls how queued steering messages are drained. */
	set steeringMode(mode: QueueMode) {
		this.steeringQueue.mode = mode;
	}

	get steeringMode(): QueueMode {
		return this.steeringQueue.mode;
	}

	/** Controls how queued follow-up messages are drained. */
	set followUpMode(mode: QueueMode) {
		this.followUpQueue.mode = mode;
	}

	get followUpMode(): QueueMode {
		return this.followUpQueue.mode;
	}

	/**
	 * EN: Queue input for a steering poll during the current run. It does not cancel a provider request or skip
	 * the current tool batch. Queue mode decides how many messages a poll consumes.
	 *
	 * ZH: 将输入放入当前运行的 steering 队列，等待调度点读取。它不会取消正在进行的模型请求，也不会跳过当前工具批次。每次消费多少消息由队列模式决定。
	 */
	steer(message: AgentMessage): void {
		this.steeringQueue.enqueue(message);
	}

	/**
	 * EN: Queue input for the point where the agent would otherwise stop. This preserves the current tool and
	 * steering workflow before starting the follow-up turn.
	 *
	 * ZH: 把输入排到 Agent 原本将要停止的位置再处理，使当前工具调用与 steering 流程先完成，再开始后续轮次。
	 */
	followUp(message: AgentMessage): void {
		this.followUpQueue.enqueue(message);
	}

	/** Remove all queued steering messages. */
	clearSteeringQueue(): void {
		this.steeringQueue.clear();
	}

	/** Remove all queued follow-up messages. */
	clearFollowUpQueue(): void {
		this.followUpQueue.clear();
	}

	/** Remove all queued steering and follow-up messages. */
	clearAllQueues(): void {
		this.clearSteeringQueue();
		this.clearFollowUpQueue();
	}

	/** Returns true when either queue still contains pending messages. */
	hasQueuedMessages(): boolean {
		return this.steeringQueue.hasItems() || this.followUpQueue.hasItems();
	}

	/** Preview the messages selected for the next turn without consuming them. */
	peekQueuedMessages(): AgentMessage[] {
		const steering = this.steeringQueue.peek();
		return steering.length > 0 ? steering : this.followUpQueue.peek();
	}

	/** Active abort signal for the current run, if any. */
	get signal(): AbortSignal | undefined {
		return this.activeRun?.abortController.signal;
	}

	/**
	 * EN: Signal cancellation to the active run. This does not synchronously clear state; providers, tools, and
	 * listeners must cooperate. Await `waitForIdle()` to observe completed cleanup.
	 *
	 * ZH: 向活动运行发送取消信号。此调用不会同步清空状态，Provider、工具及监听器需要配合取消。等待 waitForIdle() 才能确认清理完成。
	 */
	abort(): void {
		this.activeRun?.abortController.abort();
	}

	/**
	 * EN: Wait for the current run, its awaited event listeners, and final cleanup. If no run is active,
	 * resolve immediately. This is a lifecycle barrier, not a subscription to future runs.
	 *
	 * ZH: 等待当前运行、被等待的事件监听器和最终清理完成。没有活动运行时立即完成。它是当前生命周期的等待点，不会订阅未来的运行。
	 */
	waitForIdle(): Promise<void> {
		return this.activeRun?.promise ?? Promise.resolve();
	}

	/**
	 * EN: Clear conversation and queued input while preserving the replayed system prompt and tool
	 * declarations. Reject while a run is active so a live loop cannot append to reset state.
	 *
	 * ZH: 清空会话和排队输入，但保留由历史重放得到的 system 提示词与工具声明。活动运行期间拒绝重置，避免运行中的循环继续向已清空的状态追加消息。
	 */
	reset(): void {
		if (this.activeRun) {
			throw new Error("Agent is already processing. Wait for completion before resetting.");
		}

		const baseline = getCurrentSystemMessage(this._state.messages);
		this._state.messages = baseline ? [baseline] : [];
		this._state.isStreaming = false;
		this._state.streamingMessage = undefined;
		this._state.pendingToolCalls = new Set<string>();
		this._state.errorMessage = undefined;
		this.clearFollowUpQueue();
		this.clearSteeringQueue();
	}

	/**
	 * EN: Start a run from one message or a batch. Messages enter the transcript through loop events. The
	 * promise settles after the run and all awaited listeners finish.
	 *
	 * ZH: 从一条或一组消息开始运行。消息通过循环事件进入对话历史。返回的 Promise 会等待运行及所有被等待的监听器结束。
	 *
	 * EN: An active run rejects a second prompt. Use `steer()` or `followUp()` to enqueue input. Provider and
	 * tool failures normally appear as events and state; invalid lifecycle use can reject.
	 *
	 * ZH: 存在活动运行时，第二个 prompt 会被拒绝；追加输入应使用 steer() 或 followUp()。Provider 和工具失败通常通过事件与状态表达，生命周期使用错误则可能导致 Promise
	 * 拒绝。
	 */
	async prompt(message: AgentMessage | AgentMessage[]): Promise<void>;
	/**
	 * EN: Wrap text and optional images in one timestamped user message, then run the same lifecycle as the
	 * message overload. Images are appended only for this string-input form.
	 *
	 * ZH: 把文本和可选图片包装为一条带时间戳的 user 消息，然后进入消息重载使用的同一生命周期。只有字符串输入形式会追加 images 参数。
	 */
	async prompt(input: string, images?: ImageContent[]): Promise<void>;
	async prompt(input: string | AgentMessage | AgentMessage[], images?: ImageContent[]): Promise<void> {
		if (this.activeRun) {
			throw new Error(
				"Agent is already processing a prompt. Use steer() or followUp() to queue messages, or wait for completion.",
			);
		}
		const messages = this.normalizePromptInput(input, images);
		await this.runPromptMessages(messages);
	}

	/**
	 * EN: Resume an existing transcript without repeating the last input. Empty or system-only histories
	 * reject. A non-assistant tail goes to the continuation loop.
	 *
	 * ZH: 从已有对话历史继续运行，不重复追加最后一次输入。空历史或仅含 system 的历史会被拒绝；末尾不是 assistant 时进入续跑循环。
	 *
	 * EN: An assistant tail needs new queued input: consume one steering batch first, then a follow-up batch.
	 * If neither exists, reject. The steering path skips its next initial poll to preserve one-at-a-time
	 * delivery.
	 *
	 * ZH: 末尾是 assistant 时需要新的排队输入：优先取一批 steering，否则取一批 follow-up；两者都没有则拒绝。steering 路径会跳过下一次初始轮询，以维持逐条投递语义。
	 */
	async continue(): Promise<void> {
		if (this.activeRun) {
			throw new Error("Agent is already processing. Wait for completion before continuing.");
		}

		const lastMessage = this._state.messages[this._state.messages.length - 1];
		if (!lastMessage || this._state.messages.every((message) => message.role === "system")) {
			throw new Error("No messages to continue from");
		}

		if (lastMessage.role === "assistant") {
			const queuedSteering = this.steeringQueue.drain();
			if (queuedSteering.length > 0) {
				await this.runPromptMessages(queuedSteering, { skipInitialSteeringPoll: true });
				return;
			}

			const queuedFollowUps = this.followUpQueue.drain();
			if (queuedFollowUps.length > 0) {
				await this.runPromptMessages(queuedFollowUps);
				return;
			}

			throw new Error("Cannot continue from message role: assistant");
		}

		await this.runContinuation();
	}

	/**
	 * EN: Normalize input shape without validating message contents. Return an array input unchanged, wrap a
	 * single message, or create one user message from text and images. Only string input creates a timestamp.
	 *
	 * ZH: 统一输入形状，但不验证消息内容。数组直接返回，单条消息包装成数组，字符串和图片则构造成一条 user 消息。只有字符串输入会新建时间戳。
	 */
	private normalizePromptInput(
		input: string | AgentMessage | AgentMessage[],
		images?: ImageContent[],
	): AgentMessage[] {
		if (Array.isArray(input)) {
			return input;
		}

		if (typeof input !== "string") {
			return [input];
		}

		const content: Array<TextContent | ImageContent> = [{ type: "text", text: input }];
		if (images && images.length > 0) {
			content.push(...images);
		}
		return [{ role: "user", content, timestamp: Date.now() }];
	}

	/**
	 * EN: Connect the public prompt to the loop. Build a context snapshot and runtime configuration, then route
	 * each awaited loop event through `processEvents()` under one lifecycle guard.
	 *
	 * ZH: 把公开 prompt 接入循环。在统一生命周期保护下创建上下文快照和运行配置，再把循环中每个被等待的事件交给 processEvents()。
	 */
	private async runPromptMessages(
		messages: AgentMessage[],
		options: { skipInitialSteeringPoll?: boolean } = {},
	): Promise<void> {
		await this.runWithLifecycle(async (signal) => {
			await runAgentLoop(
				messages,
				this.createContextSnapshot(),
				this.createLoopConfig(options),
				(event) => this.processEvents(event),
				signal,
				this.streamFunction,
			);
		});
	}

	/**
	 * EN: Use the same lifecycle and event reducer as a new prompt, but start from the existing context without
	 * initial prompt messages.
	 *
	 * ZH: 复用新 prompt 的生命周期与事件状态处理，但从已有上下文开始，不追加初始 prompt 消息。
	 */
	private async runContinuation(): Promise<void> {
		await this.runWithLifecycle(async (signal) => {
			await runAgentLoopContinue(
				this.createContextSnapshot(),
				this.createLoopConfig(),
				(event) => this.processEvents(event),
				signal,
				this.streamFunction,
			);
		});
	}

	/**
	 * EN: Copy the transcript and executable tool arrays for the loop. Message and tool objects remain shared;
	 * this is structural separation of the outer arrays, not a deep snapshot.
	 *
	 * ZH: 为循环复制对话历史和可执行工具数组。消息与工具对象仍共享；这里只分离外层数组，并非深层快照。
	 */
	private createContextSnapshot(): AgentContext {
		return {
			messages: this._state.messages.slice(),
			tools: this._state.tools.slice(),
		};
	}

	/**
	 * EN: Capture model and hook settings for the run, and expose queue-draining callbacks. The one-use
	 * steering skip prevents continuation from consuming two batches before its first response.
	 *
	 * ZH: 为本次运行收集模型与 hook 配置，并提供消费队列的回调。只生效一次的 steering 跳过标记，避免续跑在第一次响应前消费两批输入。
	 */
	private createLoopConfig(options: { skipInitialSteeringPoll?: boolean } = {}): AgentLoopConfig {
		let skipInitialSteeringPoll = options.skipInitialSteeringPoll === true;
		return {
			model: this._state.model,
			reasoning: this._state.thinkingLevel === "off" ? undefined : this._state.thinkingLevel,
			sessionId: this.sessionId,
			onPayload: this.onPayload,
			onResponse: this.onResponse,
			onProviderStreamEvent: this.onProviderStreamEvent,
			transport: this.transport,
			thinkingBudgets: this.thinkingBudgets,
			maxRetryDelayMs: this.maxRetryDelayMs,
			toolExecution: this.toolExecution,
			beforeToolCall: this.beforeToolCall,
			afterToolCall: this.afterToolCall,
			finishTurn: this.finishTurn,
			prepareRequest: this.prepareRequest,
			prepareNextTurn:
				this.prepareNextTurnWithContext || this.prepareNextTurn
					? async (context) => {
							if (this.prepareNextTurnWithContext) {
								return await this.prepareNextTurnWithContext(context, this.signal);
							}
							return await this.prepareNextTurn?.(this.signal);
						}
					: undefined,
			convertToLlm: this.convertToLlm,
			transformContext: this.transformContext,
			getApiKey: this.getApiKey,
			getSteeringMessages: async () => {
				if (skipInitialSteeringPoll) {
					skipInitialSteeringPoll = false;
					return [];
				}
				return this.steeringQueue.drain();
			},
			getFollowUpMessages: async () => this.followUpQueue.drain(),
		};
	}

	/**
	 * EN: Own the active-run promise and AbortController. Mark state as streaming before execution, convert
	 * caught failures into assistant lifecycle events, and always release runtime state in `finally`.
	 *
	 * ZH: 持有活动运行的 Promise 与 AbortController。执行前标记 streaming，将捕获的失败转换为 assistant 生命周期事件，并在 finally 中始终释放运行时状态。
	 */
	private async runWithLifecycle(executor: (signal: AbortSignal) => Promise<void>): Promise<void> {
		if (this.activeRun) {
			throw new Error("Agent is already processing.");
		}

		const abortController = new AbortController();
		let resolvePromise = () => {};
		const promise = new Promise<void>((resolve) => {
			resolvePromise = resolve;
		});
		this.activeRun = { promise, resolve: resolvePromise, abortController };

		this._state.isStreaming = true;
		this._state.streamingMessage = undefined;
		this._state.errorMessage = undefined;

		try {
			await executor(abortController.signal);
		} catch (error) {
			await this.handleRunFailure(error, abortController.signal.aborted);
		} finally {
			this.finishRun();
		}
	}

	/**
	 * EN: Represent a thrown loop failure as a synthetic assistant message with error or aborted status,
	 * followed by turn and agent end events. This uses the same reducer as normal responses.
	 *
	 * ZH: 把循环抛出的失败表示为 error 或 aborted 状态的合成 assistant 消息，随后发出轮次与 Agent 结束事件。这些事件复用正常响应的状态处理路径。
	 */
	private async handleRunFailure(error: unknown, aborted: boolean): Promise<void> {
		const failureMessage = {
			role: "assistant",
			content: [{ type: "text", text: "" }],
			api: this._state.model.api,
			provider: this._state.model.provider,
			model: this._state.model.id,
			usage: EMPTY_USAGE,
			stopReason: aborted ? "aborted" : "error",
			errorMessage: error instanceof Error ? error.message : String(error),
			timestamp: Date.now(),
		} satisfies AgentMessage;
		await this.processEvents({ type: "message_start", message: failureMessage });
		await this.processEvents({ type: "message_end", message: failureMessage });
		await this.processEvents({ type: "turn_end", message: failureMessage, toolResults: [] });
		await this.processEvents({ type: "agent_end", messages: [failureMessage] });
	}

	/**
	 * EN: Release streaming state and pending-tool tracking after execution settles. Resolve idle waiters and
	 * remove the active run; the transcript and last error remain available for inspection.
	 *
	 * ZH: 执行结束后释放流式状态和待处理工具记录，唤醒空闲等待者并移除活动运行。对话历史与最后的错误信息仍保留，供调用者检查。
	 */
	private finishRun(): void {
		this._state.isStreaming = false;
		this._state.streamingMessage = undefined;
		this._state.pendingToolCalls = new Set<string>();
		this.activeRun?.resolve();
		this.activeRun = undefined;
	}

	/**
	 * EN: Reduce one loop event into public state, then await subscribers. Only `message_end` appends a
	 * finalized message. Partial messages and pending tool ids are runtime observations.
	 *
	 * ZH: 先把一个循环事件归入公开状态，再等待订阅者。只有 message_end 会追加已完成消息；部分消息和待处理工具 ID 属于运行时观察状态。
	 *
	 * EN: The awaited reducer is a barrier: assistant state is visible before tool preflight begins.
	 * `agent_end` clears the partial message, while `finishRun()` later marks the agent idle.
	 *
	 * ZH: 被等待的状态处理形成顺序屏障：工具预检查开始前，状态里已经能看到 assistant 消息。agent_end 清除部分消息，稍后的 finishRun() 才将 Agent 标为空闲。
	 */
	private async processEvents(event: AgentEvent): Promise<void> {
		switch (event.type) {
			case "message_start":
				this._state.streamingMessage = event.message;
				break;

			case "message_update":
				this._state.streamingMessage = event.message;
				break;

			case "message_end":
				this._state.streamingMessage = undefined;
				this._state.messages.push(event.message);
				break;

			case "tool_execution_start": {
				const pendingToolCalls = new Set(this._state.pendingToolCalls);
				pendingToolCalls.add(event.toolCallId);
				this._state.pendingToolCalls = pendingToolCalls;
				break;
			}

			case "tool_execution_end": {
				const pendingToolCalls = new Set(this._state.pendingToolCalls);
				pendingToolCalls.delete(event.toolCallId);
				this._state.pendingToolCalls = pendingToolCalls;
				break;
			}

			case "turn_end":
				if (event.message.role === "assistant" && event.message.errorMessage) {
					this._state.errorMessage = event.message.errorMessage;
				}
				break;

			case "agent_end":
				this._state.streamingMessage = undefined;
				break;
		}

		const signal = this.activeRun?.abortController.signal;
		if (!signal) {
			throw new Error("Agent listener invoked outside active run");
		}
		for (const listener of this.listeners) {
			await listener(event, signal);
		}
	}
}
