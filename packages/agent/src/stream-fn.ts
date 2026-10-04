import type { StreamFn } from "./types.ts";

let defaultStreamFn: StreamFn | undefined;

/**
 * EN: Install or clear the process-level fallback used when a caller omits streamFn at runtime. A host can
 * supply its model runtime without coupling agent-core to provider discovery. Explicit injection remains
 * the current public API.
 *
 * ZH: 安装或清除进程级流式函数回退值，在调用者运行时未提供 streamFn 时使用。宿主可接入自己的模型运行时，而无需让 agent-core 耦合 Provider 发现机制。当前公开 API
 * 仍要求显式注入。
 */
export function setDefaultStreamFn(streamFn: StreamFn | undefined): void {
	defaultStreamFn = streamFn;
}

/**
 * EN: Read the configured fallback or fail immediately with setup guidance. This selects a dependency; it
 * does not perform a model request.
 *
 * ZH: 读取已配置的回退函数，未配置时立即抛出带配置提示的错误。这里只选择依赖，不会发起模型请求。
 */
export function getDefaultStreamFn(): StreamFn {
	if (!defaultStreamFn) {
		throw new Error("No default stream function configured. Pass streamFn explicitly or call setDefaultStreamFn().");
	}
	return defaultStreamFn;
}
