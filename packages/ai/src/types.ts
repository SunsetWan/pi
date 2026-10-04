import type { TelemetryContext } from "@earendil-works/pi-telemetry";
import type { AnthropicOptions } from "./api/anthropic-messages.ts";
import type { AzureOpenAIResponsesOptions } from "./api/azure-openai-responses.ts";
import type { BedrockOptions } from "./api/bedrock-converse-stream.ts";
import type { GoogleOptions } from "./api/google-generative-ai.ts";
import type { GoogleVertexOptions } from "./api/google-vertex.ts";
import type { MistralOptions } from "./api/mistral-conversations.ts";
import type { OpenAICodexResponsesOptions } from "./api/openai-codex-responses.ts";
import type { OpenAICompletionsOptions } from "./api/openai-completions.ts";
import type { OpenAIResponsesOptions } from "./api/openai-responses.ts";
import type { PiMessagesOptions } from "./api/pi-messages.ts";
import type { AssistantMessageDiagnostic } from "./utils/diagnostics.ts";
import type { AssistantMessageEventStream } from "./utils/event-stream.ts";

export type { AssistantMessageEventStream } from "./utils/event-stream.ts";

export type KnownApi =
	| "openai-completions"
	| "mistral-conversations"
	| "openai-responses"
	| "azure-openai-responses"
	| "openai-codex-responses"
	| "anthropic-messages"
	| "bedrock-converse-stream"
	| "google-generative-ai"
	| "google-vertex"
	| "pi-messages";

export type Api = KnownApi | (string & {});

export type KnownImageApi = "openrouter-images";

export type ImageApi = KnownImageApi | (string & {});

export type KnownClassifierApi = "typesafe-system-one" | "cloudflare-workers-ai-system-one" | "llama-cpp-classify";

export type ClassifierApi = KnownClassifierApi | (string & {});

export type KnownProvider =
	| "amazon-bedrock"
	| "ant-ling"
	| "anthropic"
	| "google"
	| "google-vertex"
	| "openai"
	| "azure-openai-responses"
	| "openai-codex"
	| "radius"
	| "typesafe"
	| "nvidia"
	| "deepseek"
	| "github-copilot"
	| "xai"
	| "groq"
	| "cerebras"
	| "openrouter"
	| "vercel-ai-gateway"
	| "zai"
	| "zai-coding-cn"
	| "mistral"
	| "minimax"
	| "minimax-cn"
	| "moonshotai"
	| "moonshotai-cn"
	| "huggingface"
	| "fireworks"
	| "together"
	| "baseten"
	| "opencode"
	| "opencode-go"
	| "kimi-coding"
	| "meta"
	| "cloudflare-workers-ai"
	| "cloudflare-ai-gateway"
	| "qwen-token-plan"
	| "qwen-token-plan-cn"
	| "qwen-token-plan-individual"
	| "xiaomi"
	| "xiaomi-token-plan-cn"
	| "xiaomi-token-plan-ams"
	| "xiaomi-token-plan-sgp";
export type ProviderId = KnownProvider | string;

export type ToolChoice = "auto" | "none";
export type ThinkingLevel = "minimal" | "low" | "medium" | "high" | "xhigh" | "max";
export type ModelThinkingLevel = "off" | ThinkingLevel;
export type ThinkingLevelMap = Partial<Record<ModelThinkingLevel, string | null>>;
export type SamplingParams = Record<string, unknown>;
export type SamplingParamsByThinkingLevel = Partial<Record<ModelThinkingLevel, SamplingParams>>;
export type ChatTemplateKwargValue =
	| string
	| number
	| boolean
	| null
	| {
			$var: "thinking.enabled" | "thinking.effort" | "thinking.budget";
			omitWhenOff?: boolean;
	  };

/** Top-level request field used to cap reasoning tokens on OpenAI-compatible servers. */
export type ThinkingTokenBudgetField = "thinking_token_budget" | "thinking_budget" | "thinking_budget_tokens";

/** Token budgets for each thinking level (token-based providers only) */
export interface ThinkingBudgets {
	minimal?: number;
	low?: number;
	medium?: number;
	high?: number;
}

// Base options all providers share
export type CacheRetention = "none" | "short" | "long";

/**
 * Best-effort prompt cache lifetime in seconds for each retention tier a request can ask for.
 * A missing tier means the lifetime is unknown; pi does not warm such caches.
 */
export type ModelPromptCache = Partial<Record<Exclude<CacheRetention, "none">, number>>;

export type Transport = "sse" | "websocket" | "websocket-cached" | "auto";

/** Provider-scoped environment overrides. Values take precedence over process.env. */
export type ProviderEnv = Record<string, string>;
export type ProviderHeaders = Record<string, string | null>;
export type FetchFunction = typeof globalThis.fetch;
export type SessionAffinityFormat = "openai" | "openai-nosession" | "openrouter";

export interface ProviderResponse {
	status: number;
	headers: Record<string, string>;
}

/** Authentication, HTTP transport, and lifecycle callbacks shared by provider requests. */
export interface ProviderRequestOptions<TModel = Model<Api>> {
	signal?: AbortSignal;
	/** Explicit parent context for telemetry produced by this logical request. */
	telemetryContext?: TelemetryContext;
	apiKey?: string;
	/**
	 * Optional fetch implementation for provider HTTP requests.
	 * Defaults to `globalThis.fetch`. Provider adapters that cannot inject a custom implementation may reject it.
	 * This does not affect WebSocket transports.
	 */
	fetch?: FetchFunction;
	/**
	 * Provider-scoped environment values. These take precedence over process.env for
	 * provider configuration such as regional settings, endpoint placeholders, and
	 * proxy variables.
	 */
	env?: ProviderEnv;
	/**
	 * Optional callback for inspecting or replacing provider payloads before sending.
	 * Return undefined to keep the payload unchanged.
	 */
	onPayload?: (payload: unknown, model: TModel) => unknown | undefined | Promise<unknown | undefined>;
	/**
	 * Optional callback invoked after an HTTP response is received.
	 */
	onResponse?: (response: ProviderResponse, model: TModel) => void | Promise<void>;
	/**
	 * Optional custom HTTP headers to include in API requests.
	 * Merged with provider defaults; caller values override default headers.
	 * On AWS Bedrock these are injected via a Smithy `build`-step middleware so
	 * they are covered by SigV4 signing; reserved headers (`x-amz-*`,
	 * `authorization`, `host`) are silently ignored to preserve SigV4 / bearer auth.
	 * A null value suppresses a provider/API default header with the same name.
	 */
	headers?: ProviderHeaders;
	/**
	 * HTTP request timeout in milliseconds for providers/SDKs that support it.
	 * For example, OpenAI and Anthropic SDK clients default to 10 minutes.
	 */
	timeoutMs?: number;
	/**
	 * Maximum retry attempts for providers/SDKs that support client-side retries.
	 * For example, OpenAI and Anthropic SDK clients default to 2.
	 */
	maxRetries?: number;
	/**
	 * Maximum delay in milliseconds to wait for a retry when the server requests a long wait.
	 * If the server's requested delay exceeds this value, the request fails immediately
	 * with an error containing the requested delay, allowing higher-level retry logic
	 * to handle it with user visibility.
	 * Default: 60000 (60 seconds). Set to 0 to disable the cap.
	 */
	maxRetryDelayMs?: number;
}

export interface StreamOptions extends ProviderRequestOptions<Model<Api>> {
	/**
	 * Optional callback invoked after an HTTP response is received and before
	 * its body stream is consumed.
	 */
	onResponse?: (response: ProviderResponse, model: Model<Api>) => void | Promise<void>;
	/**
	 * Optional observer for each parsed provider stream event before Pi normalization.
	 * Event data is adapter-owned and must be treated as read-only.
	 * Adapter support is explicit; unsupported adapters do not invoke it.
	 */
	onProviderStreamEvent?: (data: unknown, model: Model<Api>) => void | Promise<void>;
	temperature?: number;
	/**
	 * Arbitrary sampling parameters merged into the request body as-is, after the named request
	 * fields, so keys here override them. Lets custom OpenAI-compatible servers (llama.cpp, vLLM,
	 * SGLang, ...) receive parameters pi does not model, e.g. `top_p`, `top_k`, `min_p`,
	 * `repetition_penalty`. Merged over `Model.samplingParams` per key. Only applied by
	 * OpenAI-compatible adapters (completions, responses, Azure responses); other APIs ignore it.
	 */
	samplingParams?: SamplingParams;
	maxTokens?: number;
	/**
	 * Preferred transport for providers that support multiple transports.
	 * Providers that do not support this option ignore it.
	 */
	transport?: Transport;
	/**
	 * Prompt cache retention preference. Providers map this to their supported values.
	 * Default: "short".
	 */
	cacheRetention?: CacheRetention;
	/**
	 * Optional session identifier for providers that support session-based caching.
	 * Providers can use this to enable prompt caching, request routing, or other
	 * session-aware features. Ignored by providers that don't support it.
	 */
	sessionId?: string;
	/**
	 * WebSocket connect timeout in milliseconds for providers that support
	 * WebSocket transports. This covers the connection/open handshake only;
	 * stream idleness after connection uses timeoutMs.
	 */
	websocketConnectTimeoutMs?: number;
	/**
	 * Optional metadata to include in API requests.
	 * Providers extract the fields they understand and ignore the rest.
	 * For example, Anthropic uses `user_id` for abuse tracking and rate limiting.
	 */
	metadata?: Record<string, unknown>;
}

export type ProviderStreamOptions = StreamOptions & Record<string, unknown>;

export interface DeferredFetchOptions extends ProviderRequestOptions<Model<Api>> {
	/**
	 * Maximum provider long-poll duration in milliseconds.
	 * Defaults to 0, which performs one status check.
	 */
	wait?: number;
}

/** Request options for best-effort deferred-response cancellation. */
export type DeferredCancelOptions = ProviderRequestOptions<Model<Api>>;

/**
 * Maps known APIs to their full provider-specific stream option types.
 * Type-only imports from API implementation modules are erased at emit, so
 * this is tree-shake safe.
 */
export interface ApiOptionsMap {
	"anthropic-messages": AnthropicOptions;
	"openai-completions": OpenAICompletionsOptions;
	"openai-responses": OpenAIResponsesOptions;
	"openai-codex-responses": OpenAICodexResponsesOptions;
	"azure-openai-responses": AzureOpenAIResponsesOptions;
	"google-generative-ai": GoogleOptions;
	"google-vertex": GoogleVertexOptions;
	"mistral-conversations": MistralOptions;
	"bedrock-converse-stream": BedrockOptions;
	"pi-messages": PiMessagesOptions;
}

/**
 * Full stream options for an API. Known APIs resolve to their concrete option
 * type; custom API strings fall back to the generic shape.
 */
export type ApiStreamOptions<TApi extends Api> = TApi extends keyof ApiOptionsMap
	? ApiOptionsMap[TApi]
	: StreamOptions & Record<string, unknown>;

/**
 * EN: Uniform dispatch contract for chat API adapters. stream accepts API-specific settings; streamSimple
 * accepts shared settings. Both receive a normalized transcript and return the same event protocol.
 *
 * ZH: 聊天 API 适配器的统一分派契约。stream 接收 API 专用设置，streamSimple 接收通用设置；二者均接收规范化对话记录，并返回同一种事件协议。
 *
 * EN: Concrete adapters retain their own option types. Provider factories and lazy wrappers use this common
 * interface to route calls without importing every SDK eagerly.
 *
 * ZH: 具体适配器仍保留自己的配置类型。Provider 工厂和延迟加载包装器通过此公共接口路由调用，无需提前加载所有 SDK。
 */
export interface ProviderStreams {
	stream(model: Model<Api>, context: TranscriptContext, options?: StreamOptions): AssistantMessageEventStream;
	streamSimple(
		model: Model<Api>,
		context: TranscriptContext,
		options?: SimpleStreamOptions,
	): AssistantMessageEventStream;
	fetchDeferred?(
		model: Model<Api>,
		handle: DeferredHandle,
		options?: DeferredFetchOptions,
	): AssistantMessageEventStream;
	cancelDeferred?(model: Model<Api>, handle: DeferredHandle, options?: DeferredCancelOptions): Promise<void>;
}

/**
 * The uniform contract of an image-generation API implementation module:
 * every image API module under `src/api/` exports exactly `generateImages`,
 * so the module itself satisfies this interface. Lazy wrappers and
 * `createProvider({ images })` pass these around as values.
 */
export interface ProviderImages {
	generateImages(
		model: ImageModel<ImageApi>,
		context: ImagesContext,
		options?: ImagesOptions,
	): Promise<AssistantImages>;
}

/** The uniform contract implemented by classifier API modules. */
export interface ProviderClassifier {
	classify(
		model: ClassifierModel<ClassifierApi>,
		context: ClassifierContext,
		options?: ClassifierOptions,
	): Promise<ClassifierResult>;
}

export interface ClassifierOptions extends ProviderRequestOptions<ClassifierModel<ClassifierApi>> {
	/**
	 * Divides the answer logits by this value before they are normalized into probabilities.
	 * Values above 1 soften the distribution; values below 1 sharpen it. Must be positive.
	 * APIs that cannot apply it ignore it.
	 */
	temperature?: number;
}

export interface ImagesOptions extends ProviderRequestOptions<ImageModel<ImageApi>> {
	/**
	 * Optional metadata to include in API requests.
	 * Providers extract the fields they understand and ignore the rest.
	 */
	metadata?: Record<string, unknown>;
}

export type ProviderImagesOptions = ImagesOptions & Record<string, unknown>;

export interface AnthropicAllowedFallbackModel {
	provider: ProviderId;
	model: string;
	cost: ModelCost;
}

// Unified options with reasoning passed to streamSimple() and completeSimple()
/**
 * EN: Provider-neutral request settings. Adapters map the shared reasoning and tool-choice values to native
 * request fields; support still depends on the selected model and API.
 *
 * ZH: 与 Provider 无关的请求设置。适配器把通用 reasoning 和工具选择值映射到原生请求字段；具体支持能力仍由所选模型和 API 决定。
 */
export interface SimpleStreamOptions extends StreamOptions {
	/** Provider-neutral tool selection for simple requests. When omitted, adapters use provider-specific behavior. */
	toolChoice?: ToolChoice;
	reasoning?: ThinkingLevel;
	/** Ask a capable provider to return a durable handle and continue the request asynchronously. */
	deferred?: boolean | { window?: "15m" | "1h" | "24h" };
	/** Custom token budgets for thinking levels (token-based providers only) */
	thinkingBudgets?: ThinkingBudgets;
}

// Generic StreamFunction with typed options.
//
// Contract:
// - Receives a normalized transcript: the system prompt and tools live in the
//   leading system message, never on the context itself.
// - Must return an AssistantMessageEventStream.
// - Direct streamSimple() calls may throw synchronously when request auth is
//   missing. Once a stream is returned, request/model/runtime failures should
//   be encoded in that stream.
// - Error termination must produce an AssistantMessage with stopReason
//   "error" or "aborted" and errorMessage, emitted via the stream protocol.
export type StreamFunction<TApi extends Api = Api, TOptions extends StreamOptions = StreamOptions> = (
	model: Model<TApi>,
	context: TranscriptContext,
	options?: TOptions,
) => AssistantMessageEventStream;

export type ImagesFunction<TOptions extends ImagesOptions = ImagesOptions> = (
	model: ImageModel<ImageApi>,
	context: ImagesContext,
	options?: TOptions,
) => Promise<AssistantImages>;

export type ClassifierFunction<TOptions extends ClassifierOptions = ClassifierOptions> = (
	model: ClassifierModel<ClassifierApi>,
	context: ClassifierContext,
	options?: TOptions,
) => Promise<ClassifierResult>;

export interface TextSignatureV1 {
	v: 1;
	id: string;
	phase?: "commentary" | "final_answer";
}

/**
 * EN: One text block in a message. textSignature preserves provider-specific replay metadata; it is not
 * text for display.
 *
 * ZH: 消息中的一个文本块。textSignature 保存 Provider 专用的重放元数据，不是用于展示的正文。
 */
export interface TextContent {
	type: "text";
	text: string;
	textSignature?: string; // e.g., for OpenAI responses, message metadata (legacy id string or TextSignatureV1 JSON)
}

/**
 * EN: Reasoning content with optional opaque replay data. A redacted block may carry only a signature;
 * preserve it for provider continuity instead of inventing visible reasoning text.
 *
 * ZH: 推理内容及可选的不透明重放数据。被隐藏的块可能只有签名，应保留它以维持 Provider 连续性，而不是自行补造可见推理文本。
 */
export interface ThinkingContent {
	type: "thinking";
	thinking: string;
	thinkingSignature?: string; // Provider-specific opaque or serialized reasoning replay data
	/** When true, the thinking content was redacted by safety filters. The opaque
	 *  encrypted payload is stored in `thinkingSignature` so it can be passed back
	 *  to the API for multi-turn continuity. */
	redacted?: boolean;
}

export interface ImageContent {
	type: "image";
	data: string; // base64 encoded image data
	mimeType: string; // e.g., "image/jpeg", "image/png"
}

/**
 * EN: Assistant request to invoke a named tool. id connects the call to its toolResult message. Streamed
 * arguments may still be partial; the agent validates finalized arguments before execution.
 *
 * ZH: assistant 请求调用命名工具的内容块。id 把调用与 toolResult 消息关联起来。流式阶段的 arguments 可能尚未完整，Agent 会在执行前验证最终参数。
 */
export interface ToolCall {
	type: "toolCall";
	id: string;
	name: string;
	arguments: JsonObject;
	thoughtSignature?: string; // Google-specific: opaque signature for reusing thought context
	/** OpenAI Responses namespace for calls to dynamically loaded or namespaced tools. */
	namespace?: string;
}

export interface Usage {
	input: number;
	output: number;
	cacheRead: number;
	cacheWrite: number;
	/** Subset of `cacheWrite` written with 1h retention. Only Anthropic reports this split. */
	cacheWrite1h?: number;
	/**
	 * Reasoning/thinking tokens, when the provider reports them. This is a subset of
	 * `output`: `output` already includes these tokens. Set to a number (possibly 0) by
	 * providers that expose a reasoning breakdown; left undefined by providers that don't.
	 */
	reasoning?: number;
	totalTokens: number;
	cost: {
		input: number;
		output: number;
		cacheRead: number;
		cacheWrite: number;
		total: number;
	};
}

/**
 * EN: Normalized response status. pending is unfinished; stop, length, and toolUse describe completed
 * generation. error and aborted are failure exits; deferred carries a handle for later retrieval.
 *
 * ZH: 规范化响应状态。pending 表示尚未完成；stop、length 和 toolUse 描述生成结束原因；error 与 aborted 表示失败退出；deferred 携带可供之后取回结果的句柄。
 */
export type StopReason = "pending" | "stop" | "length" | "toolUse" | "error" | "aborted" | "deferred";

export type JsonValue = null | boolean | number | string | readonly JsonValue[] | JsonObject;
export type JsonObject = { [key: string]: JsonValue };

type IsAny<T> = 0 extends 1 & T ? true : false;
type IsExactlyJsonValue<T> = [T] extends [JsonValue] ? ([JsonValue] extends [T] ? true : false) : false;
type IsJsonProperty<T> = IsAny<T> extends true
	? false
	: unknown extends T
		? false
		: [Exclude<T, undefined>] extends [never]
			? true
			: IsJsonCompatible<Exclude<T, undefined>>;
type InvalidJsonKeys<T extends object> = {
	[TKey in keyof T]-?: TKey extends string | number ? (IsJsonProperty<T[TKey]> extends true ? never : TKey) : TKey;
}[keyof T];
type IsJsonCompatible<T> = IsAny<T> extends true
	? false
	: unknown extends T
		? false
		: IsExactlyJsonValue<T> extends true
			? true
			: T extends null | boolean | number | string
				? true
				: T extends undefined
					? false
					: T extends readonly (infer TItem)[]
						? IsJsonCompatible<TItem>
						: T extends (...args: never[]) => unknown
							? false
							: T extends object
								? [InvalidJsonKeys<T>] extends [never]
									? true
									: false
								: false;

/** The JSON representation of a typed in-memory value. Optional object properties remain optional. */
export type JsonRepresentation<T> = IsAny<T> extends true
	? JsonValue
	: unknown extends T
		? JsonValue
		: [T] extends [JsonValue]
			? T
			: T extends readonly unknown[]
				? { [TKey in keyof T]: JsonRepresentation<Exclude<T[TKey], undefined>> }
				: T extends object
					? { [TKey in keyof T]: JsonRepresentation<Exclude<T[TKey], undefined>> }
					: never;

export interface DeferredHandle {
	provider: string;
	modelId: string;
	api: string;
	/** Provider token, such as a response id or batch id plus row id. */
	id: string;
	expiresAt?: number;
	pollAfterMs?: number;
	/** Provider conversion data required to reconstruct the final assistant message. */
	data?: JsonValue;
}

/**
 * EN: Prompt and tool state at a point in the transcript. The leading message defines the baseline. Later
 * content appends instructions, named sections replace or remove sections, and tool deltas update
 * declarations.
 *
 * ZH: 对话历史某个位置的提示词与工具状态。首条消息定义基线；后续 content 追加指令，命名 sections 替换或删除段落，工具增量更新声明。
 *
 * EN: Replay messages in order to obtain the current prompt and tools. Adapters keep these changes in place
 * when supported, otherwise collapse them into a leading system message.
 *
 * ZH: 按顺序重放消息可得到当前提示词和工具。适配器在协议支持时保留这些变化的位置，否则将它们折叠到首条 system 消息中。
 */
export interface SystemMessage {
	role: "system";
	/** Instruction text. On the leading message this is the base prompt; later, additional instructions. */
	content: string | TextContent[];
	/**
	 * Named, ordered prompt sections rendered verbatim after `content`. The leading message
	 * declares them; later messages replace sections by name, and `null` removes one. Keep
	 * each section self-delimiting (a tag, a heading) so the model can relate an update to
	 * the original. Avoid integer-like names; JSON objects reorder those.
	 */
	sections?: Record<string, string | null>;
	/** Complete definitions of tools that become available at this point. */
	toolsAdded?: Tool[];
	/** Tools that stop being available at this point. */
	toolsRemoved?: ToolReference[];
	timestamp: number; // Unix timestamp in milliseconds
}

/**
 * EN: User input with text or multimodal content and a millisecond timestamp. The agent normalizes a string
 * prompt into this role before emitting message lifecycle events.
 *
 * ZH: 包含文本或多模态内容以及毫秒时间戳的用户输入。Agent 先把字符串 prompt 规范化为该角色，再发出消息生命周期事件。
 */
export interface UserMessage {
	role: "user";
	content: string | (TextContent | ImageContent)[];
	timestamp: number; // Unix timestamp in milliseconds
}

/**
 * EN: Shared model response representation. content contains text, reasoning, and tool calls; usage and
 * stopReason describe the outcome. This shape also represents partial responses and error results.
 *
 * ZH: 统一的模型响应表示。content 包含文本、推理与工具调用，usage 和 stopReason 描述结果。部分响应和错误结果也使用这个结构。
 *
 * EN: Provider/model identifiers and signatures support tracing and replay. endTurn is diagnostic metadata
 * and does not by itself control the agent loop.
 *
 * ZH: Provider、模型标识与签名支持追踪和重放。endTurn 是诊断元数据，本身不控制 Agent 循环。
 */
export interface AssistantMessage {
	role: "assistant";
	content: (TextContent | ThinkingContent | ToolCall)[];
	api: Api;
	provider: ProviderId;
	model: string;
	responseModel?: string; // Concrete model reported by the provider when different from the requested `model`
	responseId?: string; // Provider-specific response/message identifier when the upstream API exposes one
	/** Exact provider-native effort level used for this response. Absent for legacy or unmanaged responses. */
	providerThinkingLevel?: string;
	/** Pi thinking level the agent loop requested for this response. Absent outside the agent loop and for legacy responses. */
	thinkingLevel?: ModelThinkingLevel;
	diagnostics?: AssistantMessageDiagnostic[]; // Redacted provider/runtime diagnostics for failures and recoveries.
	usage: Usage;
	stopReason: StopReason;
	deferred?: DeferredHandle;
	errorMessage?: string;
	rawStopReason?: string;
	/**
	 * Provider indication of whether the model explicitly ended its turn.
	 * Preserved for debugging and does not currently affect agent control flow.
	 */
	endTurn?: boolean;
	timestamp: number; // Unix timestamp in milliseconds
}

/** A tool call that another tool made while it ran, for example from a codemode script. */
export interface NestedToolCallRecord {
	id: string;
	name: string;
	/** Omitted when over the size limits; `argumentsBytes` then gives their size. */
	arguments?: JsonObject;
	/** UTF-8 size of the arguments as JSON, set when `arguments` is omitted. */
	argumentsBytes?: number;
	/** `unfinished`: the call was still running when the calling tool finished. */
	status: "ok" | "error" | "unfinished";
	durationMs?: number;
	/** Error text, truncated. */
	error?: string;
}

/** Bounded record of the nested calls a tool made. Results are not recorded. */
export interface NestedToolCalls {
	calls: NestedToolCallRecord[];
	/** False when calls were dropped, arguments omitted, or calls had not finished. */
	complete: boolean;
}

/**
 * EN: Transcript result associated with a toolCallId. content is model-facing, while details and nested
 * call records support the host. isError distinguishes failure from ordinary output, including text that
 * happens to describe an error.
 *
 * ZH: 通过 toolCallId 关联调用的工具结果消息。content 面向模型，details 和嵌套调用记录服务于宿主。isError 区分失败与普通输出，不能仅靠正文中出现错误描述来判断。
 */
export type ToolResultMessage<TDetails = JsonValue> = IsJsonCompatible<TDetails> extends true
	? {
			role: "toolResult";
			toolCallId: string;
			toolName: string;
			content: (TextContent | ImageContent)[]; // Supports text and images
			details?: JsonRepresentation<TDetails>;
			/** Usage from the tool execution itself, if available. Not part of main LLM context accounting. */
			usage?: Usage;
			/** Calls this tool made to other tools. Kept for the session record; not sent to the model. */
			nestedCalls?: NestedToolCalls;
			isError: boolean;
			timestamp: number; // Unix timestamp in milliseconds
		}
	: never;

/**
 * EN: Closed set of model-facing roles: system, user, assistant, and toolResult. Agent Core adds
 * application roles separately and converts them before crossing this boundary.
 *
 * ZH: 模型侧的封闭角色集合：system、user、assistant、toolResult。Agent Core 在另一层扩展应用角色，并在跨越此边界前转换它们。
 */
export type Message = SystemMessage | UserMessage | AssistantMessage | ToolResultMessage;

export type ImagesInputContent = TextContent | ImageContent;
export type ImagesOutputContent = TextContent | ImageContent;

export interface ImagesContext {
	input: ImagesInputContent[];
}

export type ImagesStopReason = "stop" | "error" | "aborted";

export interface AssistantImages {
	api: ImageApi;
	provider: ProviderId;
	model: string;
	output: ImagesOutputContent[];
	responseId?: string;
	usage?: Usage;
	stopReason: ImagesStopReason;
	errorMessage?: string;
	timestamp: number; // Unix timestamp in milliseconds
}

export interface ClassifierChoiceQuestion {
	type: "choice";
	instructions: string;
	criteria: Record<string, string>;
}

export interface ClassifierScoreQuestion {
	type: "score";
	instructions: string;
	criteria: string[];
}

export interface ClassifierBoolQuestion {
	type: "bool";
	instructions: string;
	criteria: { true: string; false: string };
}

export type ClassifierQuestion = ClassifierChoiceQuestion | ClassifierScoreQuestion | ClassifierBoolQuestion;

export interface ClassifierContext {
	state: JsonObject;
	questions: Record<string, ClassifierQuestion>;
}

export interface ClassifierChoiceAnswer {
	type: "choice";
	choice: string;
	probabilities: Record<string, number>;
	confidence: number;
}

export interface ClassifierScoreAnswer {
	type: "score";
	score: number;
	confidence: number;
}

export interface ClassifierBoolAnswer {
	type: "bool";
	probability: number;
}

export type ClassifierAnswer = ClassifierChoiceAnswer | ClassifierScoreAnswer | ClassifierBoolAnswer;
export type ClassifierStopReason = "stop" | "error" | "aborted";

export interface ClassifierResult {
	api: ClassifierApi;
	provider: ProviderId;
	model: string;
	answers: Record<string, ClassifierAnswer>;
	/** Token usage and its cost at the model's catalog price, when the service reports token counts. */
	usage?: Usage;
	stopReason: ClassifierStopReason;
	errorMessage?: string;
	timestamp: number; // Unix timestamp in milliseconds
}

import type { TSchema } from "typebox";

/** OpenAI grammar variants for constrained sampling. */
export type GrammarFormat = "openai_lark" | "openai_regex";

export type GrammarVariants = Partial<Record<GrammarFormat, string>>;

/**
 * Optional provider-side constrained sampling configs for a tool.
 *
 * The `json_schema` value roughly maps to the concept of `strict` in APIs which is
 * implemented as json-schema constrained sampling by APIs. Grammar variants let
 * callers provide provider-specific encodings of the same intended language.
 */
export type ConstrainedSamplingConfig =
	| {
			type: "json_schema";
			strict: "prefer" | "require";
	  }
	| {
			type: "grammar";
			variants: GrammarVariants;
	  };

/**
 * EN: Model-visible declaration: name, description, input schema, and optional constrained sampling. It
 * contains no execute function. AgentTool adds the host implementation at the agent layer.
 *
 * ZH: 模型可见的工具声明：名称、描述、输入 schema 和可选约束采样设置。这里没有 execute 函数，AgentTool 在 Agent 层补上宿主执行实现。
 */
export interface Tool<TParameters extends TSchema = TSchema> {
	name: string;
	description: string;
	parameters: TParameters;
	constrainedSampling?: false | ConstrainedSamplingConfig;
}

export interface ToolReference {
	name: string;
}

/**
 * EN: Convenience input accepted by Models. systemPrompt and tools are shorthand fields; normalizeContext
 * folds them into a leading system message before provider dispatch.
 *
 * ZH: Models 接收的便捷输入。systemPrompt 和 tools 是简写字段；normalizeContext 会在分派给 Provider 前将它们合入首条 system 消息。
 */
export interface Context {
	systemPrompt?: string;
	messages: Message[];
	tools?: Tool[];
}

declare const transcriptContextBrand: unique symbol;

/**
 * EN: Provider-facing context produced by normalizeContext. System messages carry the prompt and tool
 * declarations. The unique-symbol brand prevents a plain Context from being passed here accidentally at
 * compile time.
 *
 * ZH: 由 normalizeContext 产生、面向 Provider 的上下文。system 消息携带提示词和工具声明。unique symbol 品牌类型在编译期防止把普通 Context
 * 意外传到此处。
 */
export type TranscriptContext = {
	messages: Message[];
	readonly [transcriptContextBrand]: true;
};

/**
 * EN: Streaming protocol shared by providers. A successful request emits start, block start/delta/end
 * events, and done. Setup may fail with error before start; failures after start also end with error.
 * Updates and done cannot precede start.
 *
 * ZH: Provider 共享的流式协议。成功请求依次发出 start、内容块 start/delta/end 及 done。准备阶段可以在 start 前直接 error；开始后的失败也以 error
 * 结束。更新和 done 不能早于 start。
 *
 * EN: partial is a live response object, not a historical snapshot. Text/thinking grow through deltas until
 * their authoritative end event; redacted thinking can start complete. Initial tool arguments vary by
 * provider. Direct adapter streamSimple calls may throw for missing auth.
 *
 * ZH: partial 是实时响应对象，而不是历史快照。文本和推理通过 delta 增长，直到权威的 end 事件；被隐藏的推理可在开始时已完整。工具初始参数因 Provider 而异。直接调用适配器
 * streamSimple 时，认证缺失可能同步抛错。
 */
export type AssistantMessageEvent =
	| { type: "start"; partial: AssistantMessage }
	| { type: "text_start"; contentIndex: number; partial: AssistantMessage }
	| { type: "text_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
	| { type: "text_end"; contentIndex: number; content: string; partial: AssistantMessage }
	| { type: "thinking_start"; contentIndex: number; partial: AssistantMessage }
	| { type: "thinking_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
	| { type: "thinking_end"; contentIndex: number; content: string; partial: AssistantMessage }
	| { type: "toolcall_start"; contentIndex: number; partial: AssistantMessage }
	| { type: "toolcall_delta"; contentIndex: number; delta: string; partial: AssistantMessage }
	| { type: "toolcall_end"; contentIndex: number; toolCall: ToolCall; partial: AssistantMessage }
	| {
			type: "done";
			reason: Extract<StopReason, "stop" | "length" | "toolUse" | "deferred">;
			message: AssistantMessage;
	  }
	| { type: "error"; reason: Extract<StopReason, "aborted" | "error">; error: AssistantMessage };

/**
 * Compatibility settings for OpenAI-compatible completions APIs.
 * Use this to override URL-based auto-detection for custom providers.
 */
export interface OpenAICompletionsCompat {
	/** Whether the provider supports the `store` field. Default: auto-detected from URL. */
	supportsStore?: boolean;
	/** Whether the provider supports the `developer` role (vs `system`). Default: auto-detected from URL. */
	supportsDeveloperRole?: boolean;
	/** Whether the provider supports `reasoning_effort`. Default: auto-detected from URL. */
	supportsReasoningEffort?: boolean;
	/** Whether the provider supports `stream_options: { include_usage: true }` for token usage in streaming responses. Default: true. */
	supportsUsageInStreaming?: boolean;
	/** Whether streamed responses include `finish_reason`. When false, pi infers `stop` or `toolUse` when the stream ends. Default: true. */
	supportsFinishReason?: boolean;
	/** Which field to use for max tokens. Default: auto-detected from URL. */
	maxTokensField?: "max_completion_tokens" | "max_tokens";
	/** Whether tool results require the `name` field. Default: auto-detected from URL. */
	requiresToolResultName?: boolean;
	/** Whether a user message after tool results requires an assistant message in between. Default: auto-detected from URL. */
	requiresAssistantAfterToolResult?: boolean;
	/** Whether thinking blocks must be converted to text blocks with <thinking> delimiters. Default: auto-detected from URL. */
	requiresThinkingAsText?: boolean;
	/** Whether all replayed assistant messages must include an empty reasoning_content field when reasoning is enabled. Default: auto-detected from URL. */
	requiresReasoningContentOnAssistantMessages?: boolean;
	/** Format for reasoning/thinking parameter. "openai" uses reasoning_effort, "openrouter" uses reasoning: { effort }, "deepseek" uses thinking: { type } plus reasoning_effort when supported, "together" uses reasoning: { enabled } plus reasoning_effort when supported, "baseten" uses configurable chat_template_args plus reasoning_effort when supported, "zai" uses thinking: { type }, "qwen" uses top-level enable_thinking: boolean, "qwen-chat-template" uses chat_template_kwargs.enable_thinking and preserve_thinking, "chat-template" uses configurable chat_template_kwargs, "string-thinking" uses top-level thinking: string, and "ant-ling" uses reasoning: { effort } only when the mapped effort is non-null. Default: "openai". */
	thinkingFormat?:
		| "openai"
		| "openrouter"
		| "deepseek"
		| "together"
		| "baseten"
		| "zai"
		| "qwen"
		| "chat-template"
		| "qwen-chat-template"
		| "string-thinking"
		| "ant-ling";
	/** Kwargs to send as `chat_template_kwargs` when `thinkingFormat` is `chat-template`. Use `{ "$var": "thinking.enabled" }`, `{ "$var": "thinking.effort" }`, or `{ "$var": "thinking.budget" }` for pi-controlled thinking values. */
	chatTemplateKwargs?: Record<string, ChatTemplateKwargValue>;
	/** Arguments to send as `chat_template_args` when `thinkingFormat` is `baseten`. Use `{ "$var": "thinking.enabled" }`, `{ "$var": "thinking.effort" }`, or `{ "$var": "thinking.budget" }` for pi-controlled thinking values. */
	chatTemplateArgs?: Record<string, ChatTemplateKwargValue>;
	/** OpenRouter-compatible routing preferences sent as the `provider` request field. */
	openRouterRouting?: OpenRouterRouting;
	/** Vercel AI Gateway routing preferences. Only used when baseUrl points to Vercel AI Gateway. */
	vercelGatewayRouting?: VercelGatewayRouting;
	/** Whether z.ai supports top-level `tool_stream: true` for streaming tool call deltas. Default: false. */
	zaiToolStream?: boolean;
	/**
	 * Top-level request field used to cap reasoning tokens from `thinkingBudgets`.
	 * Reasoning and the answer share `max_tokens` on these endpoints, so without a budget a
	 * reasoning-heavy turn can consume the whole response and emit no answer.
	 * `"thinking_token_budget"` is vLLM, `"thinking_budget"` is Qwen/DashScope/SGLang,
	 * `"thinking_budget_tokens"` is llama.cpp. Off by default; not set on the generated catalog.
	 */
	thinkingTokenBudgetField?: ThinkingTokenBudgetField;
	/** Alias for `thinkingTokenBudgetField: "thinking_token_budget"` (vLLM). Prefer `thinkingTokenBudgetField`. Default: false. */
	supportsThinkingTokenBudget?: boolean;
	/** Whether the provider supports OpenAI custom tools with Lark/regex grammar formats. When false, grammar-constrained tools fall back to normal function tools. Default: false; the generated model catalog enables it for capable models. */
	supportsOpenAIGrammarTools?: boolean;
	/** Whether the exact model accepts system or developer messages after the conversation has started. When false, later system messages are folded into the leading system message. Default: false; the generated model catalog enables it for verified models. */
	supportsMidConvoSystemMessages?: boolean;
	/** Whether system messages can introduce additional tools mid-conversation. Requires `supportsMidConvoSystemMessages`. Default: false; the generated model catalog enables it for capable models. */
	supportsMidConvoToolAdditions?: boolean;
	/** Whether the provider supports the `strict` field in tool definitions. Default: false; generated capable models enable it explicitly. */
	supportsStrictMode?: boolean;
	/** Cache control convention for prompt caching. "anthropic" applies Anthropic-style `cache_control` markers to the system prompt, last tool definition, and last user, assistant, or tool-result text content. */
	cacheControlFormat?: "anthropic";
	/** Whether to send session-affinity data from `options.sessionId`. Default: true for OpenRouter endpoints, false otherwise. */
	sendSessionAffinityHeaders?: boolean;
	/** Session-affinity header format: `openai` sends `session_id`, `x-client-request-id`, and `x-session-affinity`; `openai-nosession` sends `x-client-request-id` and `x-session-affinity`; `openrouter` sends `x-session-id`. Does not affect the `prompt_cache_key` body param, which is governed by cache retention. Default: auto-detected. */
	sessionAffinityFormat?: SessionAffinityFormat;
	/** Whether the provider supports long prompt cache retention (`prompt_cache_retention: "24h"` or Anthropic-style `cache_control.ttl: "1h"`, depending on format). Default: true. */
	supportsLongCacheRetention?: boolean;
	/**
	 * vLLM scheduler priority sent as the top-level `priority` request field (lower values are
	 * handled earlier; server default 0). Only meaningful when vLLM runs with
	 * `--scheduling-policy priority`; useful for keeping background/batch work from stalling
	 * interactive sessions. Off by default; not set on the generated catalog.
	 */
	vllmPriority?: number;
}

/** Compatibility settings for OpenAI Responses APIs. */
export interface OpenAIResponsesCompat {
	/** Whether the provider supports the `developer` role (vs `system`). Default: true. */
	supportsDeveloperRole?: boolean;
	/** Whether the exact model accepts developer or system messages after the conversation has started. When false, later system messages are folded into the leading system message. Default: false; the generated model catalog enables it for verified models. */
	supportsMidConvoSystemMessages?: boolean;
	/** Session-affinity header format: `openai` sends `session_id` and `x-client-request-id`; `openai-nosession` sends `x-client-request-id`; `openrouter` sends `x-session-id`. Does not affect the `prompt_cache_key` body param, which is governed by cache retention. Default: auto-detected. */
	sessionAffinityFormat?: SessionAffinityFormat;
	/** Whether the provider supports long prompt cache retention. This uses `prompt_cache_options.ttl: "30m"` on GPT-5.6+ and `prompt_cache_retention: "24h"` on earlier models. Default: true. */
	supportsLongCacheRetention?: boolean;
	/** Whether the provider supports strict JSON-schema function tools. Defaults are API-specific; generated OpenAI models enable it explicitly. */
	supportsStrictMode?: boolean;
	/** Whether to emit OpenAI custom tools with Lark/regex grammar formats. When false, grammar-constrained tools fall back to normal function tools. Default: false; the generated model catalog enables it for capable models. */
	supportsOpenAIGrammarTools?: boolean;
	/** Whether the model supports message-anchored `additional_tools` input items. Default: false. */
	supportsAdditionalTools?: boolean;
	/** Whether the model supports client-executed tool search for transcript-anchored additions. Default: false. */
	supportsToolSearch?: boolean;
	/** Whether the model accepts `prompt_cache_options` (OpenAI GPT-5.6+ prompt caching). Older OpenAI models reject the parameter. Default: false. */
	supportsExplicitPromptCacheMode?: boolean;
	/** Whether the provider accepts the `max_output_tokens` parameter. Some Codex-protocol gateways reject it. Default: true. */
	supportsMaxOutputTokens?: boolean;
}

/** Compatibility settings for Anthropic Messages-compatible APIs. */
export interface AnthropicMessagesCompat {
	/**
	 * Whether the provider accepts per-tool `eager_input_streaming`.
	 * When false, the Anthropic provider omits `tools[].eager_input_streaming`
	 * and sends the legacy `fine-grained-tool-streaming-2025-05-14` beta header
	 * for tool-enabled requests.
	 * Default: true.
	 */
	supportsEagerToolInputStreaming?: boolean;
	/** Whether the provider supports Anthropic long cache retention (`cache_control.ttl: "1h"`). Default: true. */
	supportsLongCacheRetention?: boolean;
	/**
	 * Whether to send the `x-session-affinity` header from `options.sessionId`
	 * when caching is enabled. Required for providers like Fireworks that use
	 * session affinity for prompt cache routing (requests to the same replica
	 * maximize cache hits).
	 * Default: false.
	 */
	sendSessionAffinityHeaders?: boolean;
	/** Session-affinity format. `"openrouter"` sends `x-session-id`; when unset, sends `x-session-affinity`. */
	sessionAffinityFormat?: "openrouter";
	/**
	 * Whether the provider supports Anthropic-style `cache_control` markers on
	 * tool definitions. When false, `cache_control` is omitted from tool params.
	 * Some Anthropic-compatible providers (e.g., Fireworks) do not support this
	 * field on tools and may reject or ignore it.
	 * Default: true.
	 */
	supportsCacheControlOnTools?: boolean;
	/**
	 * Whether the model accepts the Anthropic `temperature` request field.
	 * Claude Opus 4.7+ rejects non-default temperature values.
	 * Default: true.
	 */
	supportsTemperature?: boolean;
	/**
	 * Whether to force adaptive thinking (`thinking.type: "adaptive"` plus
	 * `output_config.effort`) regardless of the model id. Built-in models that
	 * require adaptive thinking set this in generated metadata. Custom
	 * Anthropic-compatible providers can set this to `true` for any model whose
	 * upstream requires the adaptive format. Set to `false` to
	 * opt out on overridden built-in models.
	 * Default: false.
	 */
	forceAdaptiveThinking?: boolean;
	/** Whether to replay empty thinking signatures as `signature: ""` instead of converting thinking to text. Default: false. */
	allowEmptySignature?: boolean;
	/** Whether the provider supports Anthropic strict tool schemas. Default: false; generated Anthropic models enable it explicitly. */
	supportsStrictTools?: boolean;
	/** Whether the exact model transport supports effort-only system messages and thinking binding controls. Default: false. */
	supportsMidConvoEffort?: boolean;
	/** Whether the exact model accepts system-role messages inside the conversation. When false, later system messages are folded into the top-level system prompt. Default: false. */
	supportsMidConvoSystemMessages?: boolean;
	/** Whether the exact model accepts mid-conversation `tool_addition` blocks with inline tool definitions (`inline-tools-2026-09-15`) and `tool_removal` blocks. Requires `supportsMidConvoSystemMessages`. Default: false. */
	supportsMidConvoToolChanges?: boolean;
	/**
	 * Models Anthropic accepts in `fallbacks` for server-side refusal fallback,
	 * with local pricing metadata for returned fallback responses. When absent or
	 * empty, callers must omit `fallbacks`; Anthropic rejects the field for models
	 * with no permitted fallback targets.
	 */
	allowedFallbackModels?: AnthropicAllowedFallbackModel[];
}

/** Compatibility settings for Amazon Bedrock models. */
export interface BedrockCompat {
	/** Whether the model supports Bedrock strict tool schemas. Default: false. */
	supportsStrictMode?: boolean;
}

/** Compatibility settings for the Mistral chat API. */
export interface MistralConversationsCompat {
	/** Whether the exact model accepts system messages after the conversation has started. When false, later system messages are folded into the leading system message. Default: false. */
	supportsMidConvoSystemMessages?: boolean;
}

/**
 * OpenRouter provider routing preferences.
 * Controls which upstream providers OpenRouter routes requests to.
 * Sent as the `provider` field in the OpenRouter API request body.
 * @see https://openrouter.ai/docs/guides/routing/provider-selection
 */
export interface OpenRouterRouting {
	/** Whether to allow backup providers to serve requests. Default: true. */
	allow_fallbacks?: boolean;
	/** Whether to filter providers to only those that support all parameters in the request. Default: false. */
	require_parameters?: boolean;
	/** Data collection setting. "allow" (default): allow providers that may store/train on data. "deny": only use providers that don't collect user data. */
	data_collection?: "deny" | "allow";
	/** Whether to restrict routing to only ZDR (Zero Data Retention) endpoints. */
	zdr?: boolean;
	/** Whether to restrict routing to only models that allow text distillation. */
	enforce_distillable_text?: boolean;
	/** An ordered list of provider names/slugs to try in sequence, falling back to the next if unavailable. */
	order?: string[];
	/** List of provider names/slugs to exclusively allow for this request. */
	only?: string[];
	/** List of provider names/slugs to skip for this request. */
	ignore?: string[];
	/** A list of quantization levels to filter providers by (e.g., ["fp16", "bf16", "fp8", "fp6", "int8", "int4", "fp4", "fp32"]). */
	quantizations?: string[];
	/** Sorting strategy. Can be a string (e.g., "price", "throughput", "latency") or an object with `by` and `partition`. */
	sort?:
		| string
		| {
				/** The sorting metric: "price", "throughput", "latency". */
				by?: string;
				/** Partitioning strategy: "model" (default) or "none". */
				partition?: string | null;
		  };
	/** Maximum price per million tokens (USD). */
	max_price?: {
		/** Price per million prompt tokens. */
		prompt?: number | string;
		/** Price per million completion tokens. */
		completion?: number | string;
		/** Price per image. */
		image?: number | string;
		/** Price per audio unit. */
		audio?: number | string;
		/** Price per request. */
		request?: number | string;
	};
	/** Preferred minimum throughput (tokens/second). Can be a number (applies to p50) or an object with percentile-specific cutoffs. */
	preferred_min_throughput?:
		| number
		| {
				/** Minimum tokens/second at the 50th percentile. */
				p50?: number;
				/** Minimum tokens/second at the 75th percentile. */
				p75?: number;
				/** Minimum tokens/second at the 90th percentile. */
				p90?: number;
				/** Minimum tokens/second at the 99th percentile. */
				p99?: number;
		  };
	/** Preferred maximum latency (seconds). Can be a number (applies to p50) or an object with percentile-specific cutoffs. */
	preferred_max_latency?:
		| number
		| {
				/** Maximum latency in seconds at the 50th percentile. */
				p50?: number;
				/** Maximum latency in seconds at the 75th percentile. */
				p75?: number;
				/** Maximum latency in seconds at the 90th percentile. */
				p90?: number;
				/** Maximum latency in seconds at the 99th percentile. */
				p99?: number;
		  };
}

/**
 * Vercel AI Gateway routing preferences.
 * Controls which upstream providers the gateway routes requests to.
 * @see https://vercel.com/docs/ai-gateway/models-and-providers/provider-options
 */
export interface VercelGatewayRouting {
	/** List of provider slugs to exclusively use for this request (e.g., ["bedrock", "anthropic"]). */
	only?: string[];
	/** List of provider slugs to try in order (e.g., ["anthropic", "openai"]). */
	order?: string[];
}

export interface ModelCostRates {
	input: number; // $/million tokens
	output: number; // $/million tokens
	cacheRead: number; // $/million tokens
	cacheWrite: number; // $/million tokens
}

export interface ModelCostTier extends ModelCostRates {
	/** Use this tier for requests whose total input usage exceeds this token count. */
	inputTokensAbove: number;
}

export interface ModelCost extends ModelCostRates {
	/** Request-wide pricing tiers. The highest matching input threshold applies to the full request. */
	tiers?: ModelCostTier[];
}

export interface ModelImageResizeOptions {
	maxWidth?: number;
	maxHeight?: number;
	/** Maximum base64-encoded payload size in bytes. */
	maxBytes?: number;
	jpegQuality?: number;
}

export interface ModelImageInputLimits {
	/** Cache-safe resize profile applied before a new image enters conversation history. */
	resize?: ModelImageResizeOptions;
	/** Maximum images accepted in one provider message. */
	maxPerMessage?: number;
	/** Maximum images accepted across one provider request. */
	maxPerRequest?: number;
}

export interface ModelInputLimits {
	/** Maximum serialized provider request size in bytes. */
	maxRequestBytes?: number;
	images?: ModelImageInputLimits;
}

/** Fields shared by every catalog entry, regardless of what you can do with it. */
export interface BaseModel<TApi extends string> {
	id: string;
	name: string;
	api: TApi;
	provider: ProviderId;
	baseUrl: string;
	input: ("text" | "image")[];
	/** Provider input limits and cache-safe preprocessing metadata. */
	inputLimits?: ModelInputLimits;
	cost: ModelCost;
	headers?: Record<string, string>;
}

/**
 * EN: Chat model metadata used for dispatch and request limits. provider selects the registered runtime;
 * api selects its wire adapter. Thinking maps, compatibility flags, and token limits guide request
 * preparation rather than performing inference themselves.
 *
 * ZH: 用于请求分派与限制的聊天模型元数据。provider 选择已注册运行时，api 选择协议适配器。思考映射、兼容标记与 token 限制指导请求准备，本身不执行推理。
 */
export interface Model<TApi extends Api> extends BaseModel<TApi> {
	/**
	 * Optional: chat is the default model type, so models without `type` are chat
	 * models. Narrow mixed model lists with `isModelType()` instead of comparing
	 * `type` directly.
	 */
	type?: "chat";
	reasoning: boolean;
	/**
	 * Maps pi thinking levels to provider/model-specific values.
	 * Missing keys use provider defaults. null marks a level as unsupported.
	 */
	thinkingLevelMap?: ThinkingLevelMap;
	/** Prompt cache lifetimes per retention tier. Unset when the provider's cache behavior is unknown. */
	promptCache?: ModelPromptCache;
	contextWindow: number;
	maxTokens: number;
	/** Default sampling parameters for this model. See {@link StreamOptions.samplingParams}; per-request keys override these. */
	samplingParams?: SamplingParams;
	/** Sampling parameter overrides selected by the effective pi thinking level. */
	samplingParamsByThinkingLevel?: SamplingParamsByThinkingLevel;
	/** Compatibility overrides for OpenAI-compatible APIs. If not set, auto-detected from baseUrl. */
	compat?: TApi extends "openai-completions"
		? OpenAICompletionsCompat
		: TApi extends "openai-responses" | "azure-openai-responses" | "openai-codex-responses"
			? OpenAIResponsesCompat
			: TApi extends "anthropic-messages"
				? AnthropicMessagesCompat
				: TApi extends "bedrock-converse-stream"
					? BedrockCompat
					: TApi extends "mistral-conversations"
						? MistralConversationsCompat
						: never;
}

/** Image-generation model: usable with `generateImages()` only. */
export interface ImageModel<TApi extends ImageApi> extends BaseModel<TApi> {
	type: "image";
	/** Output modalities. Always includes `"image"`; `"text"` means the model can also return text blocks. */
	output: ("text" | "image")[];
}

/** Structured classifier model: usable with `classify()` only. */
export interface ClassifierModel<TApi extends ClassifierApi> extends BaseModel<TApi> {
	type: "classifier";
	contextWindow: number;
}

/** Model shape for each model type. */
export interface ModelTypeMap {
	chat: Model<Api>;
	image: ImageModel<ImageApi>;
	classifier: ClassifierModel<ClassifierApi>;
}

/** What a catalog entry is for. Decides which `Models` operation accepts it. */
export type ModelType = keyof ModelTypeMap;

/** Anything a provider can list. Narrow with `isModelType()`. */
export type AnyModel = ModelTypeMap[ModelType];
