import { openAIResponsesApi } from "../api/openai-responses.lazy.ts";
import { envApiKeyAuth, lazyOAuth } from "../auth/helpers.ts";
import { loadOpenAIChatGPTOAuth } from "../auth/oauth/load.ts";
import { createProvider, type Provider } from "../models.ts";
import { OPENAI_MODELS } from "./openai.models.ts";

/**
 * EN: Assemble OpenAI catalog, API-key/OAuth auth policies, and a lazy Responses adapter. This factory
 * configures a Provider; creating it does not send a model request or perform login.
 *
 * ZH: 组装 OpenAI 模型目录、API key/OAuth 认证策略及延迟加载的 Responses 适配器。此工厂只配置 Provider，创建对象本身不会发起模型请求或执行登录。
 */
export function openaiProvider(): Provider<"openai-responses"> {
	return createProvider({
		id: "openai",
		name: "OpenAI",
		baseUrl: "https://api.openai.com/v1",
		auth: {
			apiKey: envApiKeyAuth("OpenAI API key", ["OPENAI_API_KEY"]),
			oauth: lazyOAuth({
				name: "OpenAI (ChatGPT subscription)",
				isSubscription: true,
				loginLabel: "Sign in with ChatGPT",
				load: loadOpenAIChatGPTOAuth,
			}),
		},
		models: Object.values(OPENAI_MODELS),
		api: openAIResponsesApi(),
	});
}
