/**
 * Verboseness eval runner (IO).
 *
 * Runs a fixed prompt set against a running deployment across thinking modes
 * (on / off / medium) and reports response lengths + over-budget flags via the
 * pure scorer in verboseness.ts.
 *
 * Usage:
 *   tsx src/eval/run-verboseness.ts --deployment nemotron-3-ultra-mtp
 *   tsx src/eval/run-verboseness.ts --endpoint http://192.168.44.36:8000 --model nemotron-3-ultra-mtp
 *   options: --budget <maxCompletionTokens=512> --max-tokens <700> --modes on,off,medium
 *
 * Resolves a deployment (by id or displayName) through the dgx-manager API to
 * its node IP:port, then asks /v1/models for the served name. Exits non-zero if
 * any response exceeds the token budget (so it can gate CI) unless --no-fail.
 */
import {
  parseResponseLengths,
  evaluateVerboseness,
  type ThinkMode,
  type VerbSample,
} from "./verboseness.js";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:4000";

// Prompt set spans task types so verbosity can be compared across them: a
// reasoning task, an open-ended explanation, a trivial factual (should be
// short), and a tight instruction-following task (should not be padded).
const PROMPT_SET: { id: string; prompt: string }[] = [
  { id: "reasoning", prompt: "A store sells apples at 3 for $1.20 and oranges at 5 for $2.00. If I buy 9 apples and 10 oranges, what's my total, and which fruit is cheaper per unit?" },
  { id: "explain", prompt: "What is the capital of Australia, and why isn't it Sydney?" },
  { id: "trivial", prompt: "What is 2 + 2?" },
  { id: "instruction", prompt: "List exactly three prime numbers under 20. Reply with only the numbers, comma-separated, nothing else." },
];

const MODE_KWARGS: Record<ThinkMode, Record<string, boolean>> = {
  on: { enable_thinking: true },
  off: { enable_thinking: false },
  medium: { enable_thinking: true, medium_effort: true },
};

function arg(name: string, def?: string): string | undefined {
  const i = process.argv.indexOf(`--${name}`);
  return i >= 0 && i + 1 < process.argv.length ? process.argv[i + 1] : def;
}
function flag(name: string): boolean {
  return process.argv.includes(`--${name}`);
}

async function resolveEndpoint(): Promise<{ baseUrl: string; model: string }> {
  const explicitEndpoint = arg("endpoint");
  if (explicitEndpoint) {
    const model = arg("model");
    if (!model) throw new Error("--endpoint requires --model");
    return { baseUrl: explicitEndpoint.replace(/\/$/, ""), model };
  }
  const dep = arg("deployment");
  if (!dep) throw new Error("provide --deployment <id|displayName> or --endpoint <url> --model <name>");
  const list = (await (await fetch(`${API_BASE}/api/deployments`)).json()) as Array<{
    id: string;
    displayName: string | null;
    status: string;
    port: number | null;
    node: { ipAddress: string };
  }>;
  const d = list.find((x) => x.id === dep || x.displayName === dep);
  if (!d) throw new Error(`deployment '${dep}' not found`);
  if (d.status !== "running" || !d.port) throw new Error(`deployment '${dep}' is ${d.status} (port ${d.port})`);
  const baseUrl = `http://${d.node.ipAddress}:${d.port}`;
  const models = (await (await fetch(`${baseUrl}/v1/models`)).json()) as { data: { id: string }[] };
  return { baseUrl, model: models.data[0].id };
}

async function callOne(
  baseUrl: string,
  model: string,
  prompt: string,
  mode: ThinkMode,
  maxTokens: number,
): Promise<VerbSample | null> {
  try {
    const res = await fetch(`${baseUrl}/v1/chat/completions`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model,
        messages: [{ role: "user", content: prompt }],
        temperature: 0.2,
        max_tokens: maxTokens,
        chat_template_kwargs: MODE_KWARGS[mode],
      }),
    });
    if (!res.ok) {
      console.error(`  [${mode}] HTTP ${res.status}: ${(await res.text()).slice(0, 160)}`);
      return null;
    }
    const d = (await res.json()) as {
      choices: { message: Record<string, unknown> }[];
      usage?: { completion_tokens?: number };
    };
    return { prompt, mode, lengths: parseResponseLengths(d.choices[0].message, d.usage) };
  } catch (e) {
    console.error(`  [${mode}] error: ${String(e).slice(0, 160)}`);
    return null;
  }
}

async function main() {
  const budget = parseInt(arg("budget", "512")!, 10);
  const maxTokens = parseInt(arg("max-tokens", "700")!, 10);
  const modes = (arg("modes", "on,off,medium")!.split(",") as ThinkMode[]).filter((m) =>
    (["on", "off", "medium"] as string[]).includes(m),
  );

  const { baseUrl, model } = await resolveEndpoint();
  console.log(`Verboseness eval → ${baseUrl} (model=${model})`);
  console.log(`budget=${budget} tok | max_tokens=${maxTokens} | modes=${modes.join(",")}\n`);

  const samples: VerbSample[] = [];
  for (const { id, prompt } of PROMPT_SET) {
    console.log(`# ${id}: ${prompt.slice(0, 70)}${prompt.length > 70 ? "…" : ""}`);
    for (const mode of modes) {
      const s = await callOne(baseUrl, model, prompt, mode, maxTokens);
      if (s) {
        const tag = s.lengths.completionTokens > budget ? "  ⚠ OVER" : "";
        console.log(
          `  ${mode.padEnd(6)} tokens=${String(s.lengths.completionTokens).padStart(4)}` +
            ` reasoning=${String(s.lengths.reasoningChars).padStart(5)}c` +
            ` content=${String(s.lengths.contentChars).padStart(5)}c${tag}`,
        );
        samples.push({ ...s, prompt: id });
      }
    }
  }

  const summary = evaluateVerboseness(samples, { maxCompletionTokens: budget });
  console.log("\n=== summary ===");
  console.log("mean completion tokens by mode:", summary.meanTokensByMode);
  console.log(
    "thinking overhead (mean ON / mean OFF):",
    summary.thinkingOverheadRatio === null ? "n/a" : summary.thinkingOverheadRatio.toFixed(2) + "×",
  );
  console.log(`over budget (>${budget} tok): ${summary.overBudgetCount}/${summary.verdicts.length}`);

  if (summary.overBudgetCount > 0 && !flag("no-fail")) process.exit(1);
}

main().catch((e) => {
  console.error(e);
  process.exit(2);
});
