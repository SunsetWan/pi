export const DOCUMENTATION_VARIANTS = ["without_docs", "with_docs"] as const;
export type DocumentationVariant = (typeof DOCUMENTATION_VARIANTS)[number];

export type DiscoveredEvalCase = {
	file: string;
	fullName: string;
	evalSet: string;
	caseId: string;
};

export type EvalTask = DiscoveredEvalCase & {
	variant: DocumentationVariant;
	model: string;
	runNumber: number;
};

function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

/**
 * EN: Validate a flat eval-set/case naming convention and reject duplicate case identities. Discovery must
 * describe the same cohort in both documentation variants before execution is planned.
 *
 * ZH: 验证平面的评测集与案例命名约定，并拒绝重复案例标识。规划执行前，两种文档变体必须发现相同的案例集合。
 */
export function parseDiscoveredCases(value: unknown): DiscoveredEvalCase[] {
	if (!Array.isArray(value)) throw new TypeError("Discovered eval cases must be an array.");
	const identities = new Set<string>();
	return value.map((item) => {
		if (!isRecord(item) || typeof item.name !== "string" || typeof item.file !== "string") {
			throw new TypeError("Discovered eval case is invalid.");
		}
		const [evalSet, caseId, ...extra] = item.name.split(" > ");
		if (!evalSet?.trim() || !caseId?.trim() || extra.length > 0) {
			throw new TypeError(`Documentation eval must use "<eval set> > <case>": ${item.name}`);
		}
		const identity = JSON.stringify([evalSet, caseId]);
		if (identities.has(identity)) throw new TypeError(`Duplicate eval case identity: ${item.name}`);
		identities.add(identity);
		return { file: item.file, fullName: item.name, evalSet, caseId };
	});
}

/**
 * EN: Expand the fixed cohort into case, model, repetition, and documentation-variant arms. Alternate
 * variant order by repetition to reduce order bias; the plan is recorded before any arm runs.
 *
 * ZH: 把固定案例集合展开为案例、模型、重复次数和文档变体组合。按重复轮次交替变体顺序以减少顺序偏差；任一实验分支运行前先记录完整计划。
 */
export function createTaskPlan(
	cases: readonly DiscoveredEvalCase[],
	model: string,
	runsPerVariant: number,
): EvalTask[] {
	if (!model.includes("/") || model.startsWith("/") || model.endsWith("/")) {
		throw new TypeError("Model identity must contain a provider and model.");
	}
	if (!Number.isSafeInteger(runsPerVariant) || runsPerVariant < 1) {
		throw new TypeError("Runs per variant must be a positive integer.");
	}
	const tasks: EvalTask[] = [];
	for (const evalCase of cases) {
		for (let runNumber = 1; runNumber <= runsPerVariant; runNumber += 1) {
			const variants: DocumentationVariant[] =
				runNumber % 2 === 1 ? ["without_docs", "with_docs"] : ["with_docs", "without_docs"];
			for (const variant of variants) tasks.push({ ...evalCase, variant, model, runNumber });
		}
	}
	return tasks;
}
