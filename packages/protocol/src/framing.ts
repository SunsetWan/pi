const FRAME_HEADER_LENGTH = 4;
const MAX_UINT32 = 0xffff_ffff;
const PAYLOAD_BLOCK_SIZE = 64 * 1024;

/** Default upper bound for one framed CBOR payload. */
export const DEFAULT_MAX_FRAME_LENGTH = 16 * 1024 * 1024;

export interface FrameDecoderOptions {
	maxFrameLength?: number;
}

export class FrameError extends Error {
	constructor(message: string) {
		super(message);
		this.name = "FrameError";
	}
}

function resolveMaxFrameLength(options: FrameDecoderOptions | undefined): number {
	const value = options?.maxFrameLength ?? DEFAULT_MAX_FRAME_LENGTH;
	if (!Number.isSafeInteger(value) || value < 0 || value > MAX_UINT32) {
		throw new RangeError(`maxFrameLength must be an integer between 0 and ${MAX_UINT32}`);
	}
	return value;
}

/**
 * EN: Prefix payload bytes with an unsigned four-byte big-endian length. This framing function accepts an
 * empty payload; a higher-level codec can impose stricter content rules.
 *
 * ZH: 用无符号四字节大端长度作为负载前缀。帧层允许空负载；更高层的编解码器可以施加更严格的内容限制。
 */
export function encodeFrame(payload: Uint8Array): Uint8Array {
	if (!(payload instanceof Uint8Array)) throw new TypeError("Frame payload must be a Uint8Array");
	if (payload.byteLength > MAX_UINT32) throw new RangeError("Frame payload exceeds the unsigned 32-bit length limit");
	const frame = new Uint8Array(FRAME_HEADER_LENGTH + payload.byteLength);
	const length = payload.byteLength;
	frame[0] = length >>> 24;
	frame[1] = length >>> 16;
	frame[2] = length >>> 8;
	frame[3] = length;
	frame.set(payload, FRAME_HEADER_LENGTH);
	return frame;
}

type DecoderState = "open" | "ended" | "failed";

/**
 * EN: Reassemble frames across arbitrary chunk boundaries with bounded block allocation. Read the header
 * before allocating payload storage, reject oversized declarations early, and copy input bytes so caller
 * mutation cannot change buffered data.
 *
 * ZH: 跨任意输入分块重新组装帧，并按固定大小块分配存储。先读帧头，在分配负载前拒绝超限长度；复制输入字节，避免调用者修改已缓冲的数据。
 */
export class FrameDecoder {
	private readonly header = new Uint8Array(FRAME_HEADER_LENGTH);
	private headerLength = 0;
	private readonly maxFrameLength: number;
	private payloadBlocks: Uint8Array[] = [];
	private currentPayloadBlock: Uint8Array | undefined;
	private currentPayloadBlockLength = 0;
	private expectedPayloadLength: number | undefined;
	private payloadLength = 0;
	private state: DecoderState = "open";

	constructor(options?: FrameDecoderOptions) {
		this.maxFrameLength = resolveMaxFrameLength(options);
	}

	/**
	 * EN: Consume fragmented or coalesced bytes and return complete payloads in order. Incomplete header and
	 * payload state stay in this decoder until another push or end.
	 *
	 * ZH: 消费拆分或合并的字节块，按顺序返回完整负载。不完整帧头和负载保存在当前解码器中，等待下一次 push 或 end。
	 */
	push(chunk: Uint8Array): Uint8Array[] {
		if (this.state === "ended") throw new FrameError("Frame decoder has ended");
		if (this.state === "failed") throw new FrameError("Frame decoder has failed");
		if (!(chunk instanceof Uint8Array)) throw new TypeError("Frame chunk must be a Uint8Array");

		const frames: Uint8Array[] = [];
		let chunkOffset = 0;
		while (chunkOffset < chunk.byteLength) {
			if (this.expectedPayloadLength === undefined) {
				const headerBytes = Math.min(FRAME_HEADER_LENGTH - this.headerLength, chunk.byteLength - chunkOffset);
				this.header.set(chunk.subarray(chunkOffset, chunkOffset + headerBytes), this.headerLength);
				this.headerLength += headerBytes;
				chunkOffset += headerBytes;
				if (this.headerLength < FRAME_HEADER_LENGTH) continue;

				const frameLength =
					this.header[0]! * 0x1_000_000 + this.header[1]! * 0x1_0000 + this.header[2]! * 0x100 + this.header[3]!;
				this.headerLength = 0;
				if (frameLength > this.maxFrameLength) {
					this.fail(`Frame length ${frameLength} exceeds configured limit of ${this.maxFrameLength}`);
				}
				if (frameLength === 0) {
					frames.push(new Uint8Array());
					continue;
				}
				this.expectedPayloadLength = frameLength;
				this.payloadBlocks = [];
				this.currentPayloadBlock = undefined;
				this.currentPayloadBlockLength = 0;
				this.payloadLength = 0;
			}

			const expectedPayloadLength = this.expectedPayloadLength;
			if (expectedPayloadLength === undefined) continue;
			while (chunkOffset < chunk.byteLength && this.payloadLength < expectedPayloadLength) {
				let block = this.currentPayloadBlock;
				if (!block || this.currentPayloadBlockLength === block.byteLength) {
					block = new Uint8Array(Math.min(PAYLOAD_BLOCK_SIZE, expectedPayloadLength - this.payloadLength));
					this.payloadBlocks.push(block);
					this.currentPayloadBlock = block;
					this.currentPayloadBlockLength = 0;
				}
				const payloadBytes = Math.min(
					block.byteLength - this.currentPayloadBlockLength,
					chunk.byteLength - chunkOffset,
				);
				block.set(chunk.subarray(chunkOffset, chunkOffset + payloadBytes), this.currentPayloadBlockLength);
				this.currentPayloadBlockLength += payloadBytes;
				this.payloadLength += payloadBytes;
				chunkOffset += payloadBytes;
			}
			if (this.payloadLength === expectedPayloadLength) {
				if (this.payloadBlocks.length === 1) {
					frames.push(this.payloadBlocks[0]!);
				} else {
					const payload = new Uint8Array(expectedPayloadLength);
					let offset = 0;
					for (const payloadBlock of this.payloadBlocks) {
						payload.set(payloadBlock, offset);
						offset += payloadBlock.byteLength;
					}
					frames.push(payload);
				}
				this.payloadBlocks = [];
				this.currentPayloadBlock = undefined;
				this.currentPayloadBlockLength = 0;
				this.expectedPayloadLength = undefined;
				this.payloadLength = 0;
			}
		}
		return frames;
	}

	/**
	 * EN: Mark the byte stream ended and reject any incomplete header or payload. End-of-stream is part of
	 * validation; a clean transport close does not make a truncated frame valid.
	 *
	 * ZH: 标记字节流结束，并拒绝残留的不完整帧头或负载。流结束也是验证的一部分；传输正常关闭不代表截断帧合法。
	 */
	end(): void {
		if (this.state === "ended") throw new FrameError("Frame decoder has ended");
		if (this.state === "failed") throw new FrameError("Frame decoder has failed");
		if (this.headerLength !== 0 || this.expectedPayloadLength !== undefined) {
			this.fail("Truncated frame at end of stream");
		}
		this.state = "ended";
	}

	private fail(message: string): never {
		this.state = "failed";
		this.headerLength = 0;
		this.payloadBlocks = [];
		this.currentPayloadBlock = undefined;
		this.currentPayloadBlockLength = 0;
		this.expectedPayloadLength = undefined;
		this.payloadLength = 0;
		throw new FrameError(message);
	}
}
