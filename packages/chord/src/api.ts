import { FacetKernel } from "./facets/host.ts";
import { disposeLoadedFacets } from "./facets/loader.ts";
import { RemoteServiceBindingImpl } from "./services/consumer.ts";
import { attachReplicatedStateSource, MutableReplicatedStateImpl } from "./services/state.ts";
import type {
	AttachedReplicatedState,
	Facet,
	FacetHost,
	FacetLoader,
	FacetOptions,
	LoadedFacets,
	MutableReplicatedState,
	RemoteServiceBinding,
	RemoteServiceBindingOptions,
	RemoteServiceContract,
	ReplicatedStateSource,
	ReplicatedStateSourceOptions,
	Service,
} from "./types.ts";

/**
 * EN: Activate a complete facet graph before exposing its frozen host facade. Setup declarations,
 * dependency validation, binding readiness, and activation live in FacetKernel.
 *
 * ZH: 完整 facet 图激活后才暴露冻结的 host 外观。声明收集、依赖验证、绑定就绪与激活顺序都由 FacetKernel 实现。
 */
export async function createFacetHost(options: FacetOptions): Promise<FacetHost> {
	const kernel = new FacetKernel(options);
	await kernel.activate();
	return Object.freeze({
		services: kernel.provider,
		reload: (facets: readonly Facet[]) => kernel.reload(facets),
		dispose: () => kernel.dispose(),
	});
}

export function createStaticFacetLoader(facets: readonly Facet[]): FacetLoader {
	const loadedFacets = Object.freeze([...facets]);
	return {
		async load() {
			return { facets: loadedFacets, async dispose() {} };
		},
	};
}

export function combineFacetLoaders(loaders: readonly FacetLoader[]): FacetLoader {
	return {
		async load() {
			const loaded: LoadedFacets[] = [];
			try {
				for (const loader of loaders) loaded.push(await loader.load());
			} catch (error) {
				const cleanupErrors = await disposeLoadedFacets(loaded.reverse());
				if (cleanupErrors.length > 0) {
					throw new AggregateError([error, ...cleanupErrors], "Facet loading and cleanup failed");
				}
				throw error;
			}
			let disposed = false;
			return {
				facets: Object.freeze(loaded.flatMap(({ facets }) => facets)),
				async dispose() {
					if (disposed) return;
					disposed = true;
					const errors = await disposeLoadedFacets([...loaded].reverse());
					if (errors.length === 1) throw errors[0];
					if (errors.length > 1) throw new AggregateError(errors, "Failed to dispose loaded facets");
				},
			};
		},
	};
}

export function defineFacet(facet: Facet): Facet {
	return facet;
}

export function defineService<T>(id: string, options: { readonly local: true }): Service<T>;
export function defineService<T>(
	id: string,
	...options: [RemoteServiceContract<T>] extends [never]
		? readonly [options: never]
		: readonly [options?: { readonly local?: false }]
): Service<T>;
/**
 * EN: Create a stable service token after checking its namespace. Local services permit arbitrary
 * JavaScript contracts; remotely exposed services must satisfy the separate JSON and method/state contract.
 *
 * ZH: 检查命名空间后创建稳定服务 token。本地服务允许任意 JavaScript 契约；远程暴露服务还必须满足独立的 JSON 与方法、状态契约。
 */
export function defineService(id: string, options?: { readonly local?: boolean }): Service<unknown> {
	if (id.length === 0) throw new TypeError("Service ID must not be empty");
	// TODO: check if the reserved namespace should be part of Chord.
	if (id.startsWith("$chord.")) throw new TypeError("Service IDs beginning with $chord. are reserved");
	return Object.freeze({ id, local: options?.local ?? false });
}

export function createRemoteServiceBinding(options: RemoteServiceBindingOptions): RemoteServiceBinding {
	return new RemoteServiceBindingImpl(options);
}

export function replicatedState<T>(
	source: ReplicatedStateSource<T>,
	options?: ReplicatedStateSourceOptions,
): AttachedReplicatedState<T>;
/**
 * EN: Take immutable ownership of an alias-free strict-JSON root. The object is not frozen or defensively
 * copied; callers must stop mutating transferred and published values and make changes through the draft
 * API.
 *
 * ZH: 接收无共享别名的严格 JSON 根对象，并取得不可变所有权。对象不会被冻结或防御性复制；调用者必须停止修改移交和已发布的值，改用 draft API 变更。
 */
export function replicatedState<T extends object>(initial: T): MutableReplicatedState<T>;
export function replicatedState(
	initialOrSource: object | ReplicatedStateSource<unknown>,
	options?: ReplicatedStateSourceOptions,
): MutableReplicatedState<object> | AttachedReplicatedState<unknown> {
	if (isReplicatedStateSource(initialOrSource)) return attachReplicatedStateSource(initialOrSource, options);
	return new MutableReplicatedStateImpl(initialOrSource);
}

function isReplicatedStateSource(value: object): value is ReplicatedStateSource<unknown> {
	return typeof (value as { readonly attach?: unknown }).attach === "function";
}
