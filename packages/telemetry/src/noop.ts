import type { SpanOptions, TelemetryContext, TelemetrySpan } from "./index.ts";

/**
 * EN: Run the callback synchronously once and expose its result as a Promise. Swallow no business failure:
 * a synchronous throw becomes a rejection, while span recording methods have no effect.
 *
 * ZH: 同步执行一次回调，并用 Promise 暴露结果。不吞掉业务失败：同步抛出转为拒绝，只有 span 记录方法本身不产生效果。
 */
function startNoopSpan<T>(_options: SpanOptions, callback: (span: TelemetrySpan) => T | Promise<T>): Promise<T> {
	try {
		return Promise.resolve(callback(noopTelemetrySpan));
	} catch (error) {
		return Promise.reject(error);
	}
}

const noopTelemetrySpan: TelemetrySpan = {
	startSpan: startNoopSpan,
	addEvent: () => {},
	setAttributes: () => {},
	setStatus: () => {},
};
Object.freeze(noopTelemetrySpan);

/** Shared telemetry context used when an application does not provide one. */
export const NOOP_TELEMETRY_CONTEXT: TelemetryContext = noopTelemetrySpan;
