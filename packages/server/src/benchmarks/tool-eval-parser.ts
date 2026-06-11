export type ToolEvalCategoryInput = {
  code: string;
  label: string;
  percent: number;
  earned: number;
  maxPoints: number;
  passCount: number;
  partialCount: number;
  failCount: number;
};

export type ToolEvalSummary = {
  finalScore: number;
  rating: string;
  deployability: number | null;
  responsiveness: number | null;
  totalScenarios: number;
  totalPoints: number | null;
  maxPoints: number | null;
  safetyWarnings: string[];
  categories: ToolEvalCategoryInput[];
};

type Obj = Record<string, unknown>;

function reqNum(o: Obj, key: string): number {
  const v = o[key];
  if (typeof v !== "number" || !Number.isFinite(v)) {
    throw new Error(`tool-eval result missing required numeric field: ${key}`);
  }
  return v;
}

function reqStr(o: Obj, key: string): string {
  const v = o[key];
  if (typeof v !== "string") {
    throw new Error(`tool-eval result missing required string field: ${key}`);
  }
  return v;
}

function optNum(o: Obj, key: string): number | null {
  const v = o[key];
  return typeof v === "number" && Number.isFinite(v) ? v : null;
}

// Parse the tool-eval-bench --json payload (schema_version "1", CLI v2.0.6).
// Headline fields are top-level; total/max points and the per-category
// breakdown live under `scores`. Fail-fast: anything we depend on that is
// absent or the wrong type throws, rather than silently defaulting.
export function parseToolEvalResults(jsonText: string): ToolEvalSummary {
  let parsed: Obj;
  try {
    parsed = JSON.parse(jsonText) as Obj;
  } catch (e) {
    throw new Error(`failed to parse tool-eval JSON: ${(e as Error).message}`);
  }

  const scores = parsed.scores;
  if (!scores || typeof scores !== "object") {
    throw new Error("tool-eval result missing required object: scores");
  }
  const s = scores as Obj;

  const rawCats = s.category_scores;
  if (!Array.isArray(rawCats)) {
    throw new Error("tool-eval result missing required array: scores.category_scores");
  }
  const categories: ToolEvalCategoryInput[] = rawCats.map((c) => {
    const cat = c as Obj;
    return {
      code: reqStr(cat, "category"),
      label: reqStr(cat, "label"),
      percent: reqNum(cat, "percent"),
      earned: reqNum(cat, "earned"),
      maxPoints: reqNum(cat, "max"),
      passCount: reqNum(cat, "pass_count"),
      partialCount: reqNum(cat, "partial_count"),
      failCount: reqNum(cat, "fail_count"),
    };
  });

  const rawWarnings = parsed.safety_warnings;
  const safetyWarnings = Array.isArray(rawWarnings) ? rawWarnings.map(String) : [];

  return {
    finalScore: reqNum(parsed, "final_score"),
    rating: reqStr(parsed, "rating"),
    deployability: optNum(parsed, "deployability"),
    responsiveness: optNum(parsed, "responsiveness"),
    totalScenarios: reqNum(parsed, "total_scenarios"),
    totalPoints: optNum(s, "total_points"),
    maxPoints: optNum(s, "max_points"),
    safetyWarnings,
    categories,
  };
}
