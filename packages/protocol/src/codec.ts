import { isJsonValue } from "@earendil-works/chord";
import { Check } from "typebox/value";
import { decodeCbor, encodeCbor } from "./cbor/index.ts";
import { DEFAULT_MAX_FRAME_LENGTH, encodeFrame, FrameDecoder, type FrameDecoderOptions } from "./framing.ts";
import {
	type ClientMessage,
	ClientMessageSchema,
	PROTOCOL_VERSION,
	type ServerMessage,
	ServerMessageSchema,
} from "./protocol.ts";

export class ProtocolValidationError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "ProtocolValidationError";
	}
}

/**
 * EN: Validate the strict envelope schema and the recursive JSON-value contract. A value accepted by a
 * broad schema is still rejected if it contains unsupported nested values.
 *
 * ZH: 同时验证严格信封结构与递归 JSON 值约束。即使外层 schema 接受某字段，内部含有不支持的值时仍会拒绝。
 */
export function parseClientMessage(value: unknown): ClientMessage {
	if (!Check(ClientMessageSchema, value) || !isJsonValue(value)) {
		throw new ProtocolValidationError("Invalid client protocol message");
	}
	return value;
}

/**
 * EN: Apply the same envelope and JSON-value boundary to server messages. This does not interpret Chord
 * service operations inside payloads.
 *
 * ZH: 对服务端消息应用相同的信封与 JSON 值边界；这里不解释负载内部的 Chord 服务操作。
 */
export function parseServerMessage(value: unknown): ServerMessage {
	if (!Check(ServerMessageSchema, value) || !isJsonValue(value)) {
		throw new ProtocolValidationError("Invalid server protocol message");
	}
	return value;
}

function boundedErrorMessage(error: unknown): string {
	if (!(error instanceof Error)) return "Unknown codec error";
	return error.message.length <= 500 ? error.message : `${error.message.slice(0, 497)}...`;
}

function encodeProtocolMessage<T>(
	value: T,
	parse: (candidate: unknown) => T,
	kind: string,
	options?: FrameDecoderOptions,
): Uint8Array {
	const validated = parse(value);
	try {
		const maxFrameLength = options?.maxFrameLength ?? DEFAULT_MAX_FRAME_LENGTH;
		return encodeFrame(encodeCbor(validated, { maxByteLength: maxFrameLength }));
	} catch (error) {
		if (error instanceof ProtocolValidationError) throw error;
		throw new ProtocolValidationError(`Unable to encode ${kind} protocol message: ${boundedErrorMessage(error)}`);
	}
}

/**
 * EN: Validate the message, encode CBOR, enforce the configured payload limit, then add the four-byte
 * length prefix. A typed caller must still pass runtime validation.
 *
 * ZH: 先验证消息，再编码 CBOR、检查配置的负载上限，最后添加四字节长度前缀。即使调用端有静态类型，也必须通过运行时验证。
 */
export function encodeClientMessage(message: ClientMessage, options?: FrameDecoderOptions): Uint8Array {
	return encodeProtocolMessage(message, parseClientMessage, "client", options);
}

/** Validates and encodes one complete length-prefixed server message. */
export function encodeServerMessage(message: ServerMessage, options?: FrameDecoderOptions): Uint8Array {
	return encodeProtocolMessage(message, parseServerMessage, "server", options);
}

/**
 * EN: Compose frame assembly, CBOR decoding, and envelope validation. Any decoding failure is latched, so
 * later chunks cannot resume a stream whose message boundary or meaning is no longer trusted.
 *
 * ZH: 组合帧组装、CBOR 解码与信封验证。任一解码错误都会保留为终止性失败，后续数据不能继续使用已失去可信边界或语义的流。
 */
class ValidatedMessageDecoder<T> {
	private failed = false;
	private readonly frames: FrameDecoder;
	private readonly kind: string;
	private readonly maxFrameLength: number;
	private readonly parse: (candidate: unknown) => T;

	constructor(kind: string, parse: (candidate: unknown) => T, options?: FrameDecoderOptions) {
		this.frames = new FrameDecoder(options);
		this.kind = kind;
		this.maxFrameLength = options?.maxFrameLength ?? DEFAULT_MAX_FRAME_LENGTH;
		this.parse = parse;
	}

	push(chunk: Uint8Array): T[] {
		if (this.failed) throw new ProtocolValidationError(`${this.kind} message decoder has failed`);
		try {
			const messages: T[] = [];
			for (const frame of this.frames.push(chunk)) {
				messages.push(this.parse(decodeCbor(frame, { maxByteLength: this.maxFrameLength })));
			}
			return messages;
		} catch (error) {
			this.failed = true;
			if (error instanceof ProtocolValidationError) throw error;
			throw new ProtocolValidationError(`Invalid ${this.kind} protocol frame: ${boundedErrorMessage(error)}`);
		}
	}

	end(): void {
		if (this.failed) throw new ProtocolValidationError(`${this.kind} message decoder has failed`);
		try {
			this.frames.end();
		} catch (error) {
			this.failed = true;
			throw new ProtocolValidationError(`Invalid ${this.kind} protocol framing: ${boundedErrorMessage(error)}`);
		}
	}
}

/** Incrementally decodes and validates framed client messages. */
export class ClientMessageDecoder {
	private readonly decoder: ValidatedMessageDecoder<ClientMessage>;

	constructor(options?: FrameDecoderOptions) {
		this.decoder = new ValidatedMessageDecoder("client", parseClientMessage, options);
	}

	push(chunk: Uint8Array): ClientMessage[] {
		return this.decoder.push(chunk);
	}

	end(): void {
		this.decoder.end();
	}
}

/** Incrementally decodes and validates framed server messages. */
export class ServerMessageDecoder {
	private readonly decoder: ValidatedMessageDecoder<ServerMessage>;

	constructor(options?: FrameDecoderOptions) {
		this.decoder = new ValidatedMessageDecoder("server", parseServerMessage, options);
	}

	push(chunk: Uint8Array): ServerMessage[] {
		return this.decoder.push(chunk);
	}

	end(): void {
		this.decoder.end();
	}
}

export function isSupportedProtocolVersion(version: number): version is typeof PROTOCOL_VERSION {
	return Number.isInteger(version) && version === PROTOCOL_VERSION;
}
