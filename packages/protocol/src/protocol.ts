import type { JsonValue } from "@earendil-works/chord";
import Type, { type Static } from "typebox";
import { Check } from "typebox/value";

/**
 * EN: Version of the outer routing protocol. Chord owns the grammar inside call, result, and update
 * payloads. Peers must complete the hello exchange before sending service traffic.
 *
 * ZH: 外层路由协议的版本。call、result 和 update 内部的语法由 Chord 负责；双方完成 hello 握手后才能传输服务消息。
 */
export const PROTOCOL_VERSION = 8 as const;

const IdSchema = Type.String({ minLength: 1 });
const OpaqueJsonValueSchema = Type.Unsafe<JsonValue>(Type.Unknown());
const StrictObject = <const T extends Parameters<typeof Type.Object>[0]>(properties: T) =>
	Type.Object(properties, { additionalProperties: false });

const ServerIdSchema = Type.String({
	pattern: "^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$",
});
export type ServerId = Static<typeof ServerIdSchema>;

export function isServerId(value: unknown): value is ServerId {
	return Check(ServerIdSchema, value);
}

const ProtocolErrorSchema = StrictObject({
	code: IdSchema,
	message: Type.String(),
});
export type ProtocolErrorCode = string;
export type ProtocolError = Static<typeof ProtocolErrorSchema>;

/** Must be the first frame sent by a client. */
const ClientHelloSchema = StrictObject({
	type: Type.Literal("hello"),
	version: Type.Integer({ minimum: 0 }),
});
export type ClientHello = Static<typeof ClientHelloSchema>;

/** A server-wide call, fenced to one logical server. */
const ServerTargetSchema = StrictObject({
	serverId: ServerIdSchema,
});
/** A session call, fenced to one logical server, durable session, and live attachment. */
const SessionTargetSchema = StrictObject({
	serverId: ServerIdSchema,
	sessionId: IdSchema,
	attachmentId: IdSchema,
});
/**
 * EN: Address one attachment to a session on a specific server. The attachment id distinguishes a current
 * attachment from an older attachment to the same session.
 *
 * ZH: 定位指定服务端上某次会话挂接。attachmentId 用于区分同一会话的当前挂接与旧挂接，避免旧请求落入新的挂接生命周期。
 */
export type SessionTarget = Static<typeof SessionTargetSchema>;
const RpcTargetSchema = Type.Union([ServerTargetSchema, SessionTargetSchema]);
export type RpcTarget = Static<typeof RpcTargetSchema>;

const RequestEnvelopeSchema = StrictObject({
	type: Type.Literal("request"),
	id: IdSchema,
	target: RpcTargetSchema,
	call: OpaqueJsonValueSchema,
});
const CancelEnvelopeSchema = StrictObject({
	type: Type.Literal("cancel"),
	id: IdSchema,
	target: RpcTargetSchema,
});
/**
 * EN: Carry request identity, routing target, and a JSON-compatible service call. The protocol validates
 * this envelope; the receiving service layer validates the call grammar and arguments.
 *
 * ZH: 携带请求标识、路由目标和可表示为 JSON 的服务调用。协议层验证信封，接收端服务层验证内部调用语法与参数。
 */
export type RequestEnvelope = Static<typeof RequestEnvelopeSchema>;
export type CancelEnvelope = Static<typeof CancelEnvelopeSchema>;
export const ClientMessageSchema = Type.Union([ClientHelloSchema, RequestEnvelopeSchema, CancelEnvelopeSchema]);
export type ClientMessage = Static<typeof ClientMessageSchema>;

const ServerHelloSchema = StrictObject({
	type: Type.Literal("hello"),
	version: Type.Literal(PROTOCOL_VERSION),
	serverId: ServerIdSchema,
});
const ServerHelloErrorSchema = StrictObject({
	type: Type.Literal("hello_error"),
	error: ProtocolErrorSchema,
});
const ResponseEnvelopeSchema = Type.Union([
	StrictObject({
		type: Type.Literal("response"),
		id: IdSchema,
		ok: Type.Literal(true),
		result: Type.Optional(OpaqueJsonValueSchema),
	}),
	StrictObject({
		type: Type.Literal("response"),
		id: IdSchema,
		ok: Type.Literal(false),
		error: ProtocolErrorSchema,
	}),
]);
const ServiceEventEnvelopeSchema = StrictObject({
	type: Type.Literal("service_update"),
	subscriptionId: IdSchema,
	update: OpaqueJsonValueSchema,
});
/** Out-of-band update to this presentation's selected Session route. */
const AttachmentEnvelopeSchema = StrictObject({
	type: Type.Literal("attachment"),
	attachment: Type.Union([SessionTargetSchema, Type.Null()]),
});
export const ServerMessageSchema = Type.Union([
	ServerHelloSchema,
	ServerHelloErrorSchema,
	ResponseEnvelopeSchema,
	ServiceEventEnvelopeSchema,
	AttachmentEnvelopeSchema,
]);
export type ServerHello = Static<typeof ServerHelloSchema>;
export type ServerHelloError = Static<typeof ServerHelloErrorSchema>;
export type ResponseEnvelope = Static<typeof ResponseEnvelopeSchema>;
export type ServiceEventEnvelope = Static<typeof ServiceEventEnvelopeSchema>;
export type AttachmentEnvelope = Static<typeof AttachmentEnvelopeSchema>;
export type ServerMessage = Static<typeof ServerMessageSchema>;
