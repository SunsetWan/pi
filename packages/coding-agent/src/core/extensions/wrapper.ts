/**
 * Tool wrappers for extension-registered tools.
 *
 * These wrappers only adapt tool execution so extension tools receive the runner context.
 * Tool call and tool result interception is handled by AgentSession via agent-core hooks.
 */

import type { AgentTool } from "@earendil-works/pi-agent-core";
import { wrapToolDefinition } from "../tools/tool-definition-wrapper.ts";
import type { ExtensionRunner } from "./runner.ts";
import type { RegisteredTool } from "./types.ts";

/**
 * EN: Adapt one registered definition into AgentTool and create a fresh tool context from its runner for
 * each call. Tool-call blocking and result replacement live in AgentSession hooks, not in this adapter.
 *
 * ZH: 将一个已注册定义适配为 AgentTool，并在每次调用时从 runner 创建新的工具上下文。工具调用阻止和结果替换位于 AgentSession hook 中，不在此适配器中。
 */
export function wrapRegisteredTool(registeredTool: RegisteredTool, runner: ExtensionRunner): AgentTool {
	return wrapToolDefinition(registeredTool.definition, (toolCallId, signal) =>
		runner.createToolContext(toolCallId, signal),
	);
}

/**
 * EN: Apply the same context adapter to a tool list while preserving order. Registration, name replacement,
 * and active-tool selection remain the session runtime's responsibility.
 *
 * ZH: 保持顺序，为工具列表应用相同的上下文适配。注册、同名替换和活动工具选择仍由会话运行时负责。
 */
export function wrapRegisteredTools(registeredTools: RegisteredTool[], runner: ExtensionRunner): AgentTool[] {
	return registeredTools.map((tool) => wrapRegisteredTool(tool, runner));
}
