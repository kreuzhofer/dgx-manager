export type LaunchShell = "bash" | "powershell";

/** Throwaway bearer token — vLLM ignores its value; its presence stops
 *  Claude Code from starting the interactive OAuth login flow. */
export const CLAUDE_AUTH_TOKEN = "dgx-local";

/** Ordered so the rendered snippet is deterministic. All three model-tier
 *  vars map to the one served model, so whichever tier Claude Code picks
 *  hits this deployment. */
function envPairs(baseUrl: string, model: string, authToken: string, maxModelLen?: number): [string, string][] {
  const pairs: [string, string][] = [
    ["ANTHROPIC_BASE_URL", baseUrl],
    ["ANTHROPIC_AUTH_TOKEN", authToken],
    ["ANTHROPIC_DEFAULT_OPUS_MODEL", model],
    ["ANTHROPIC_DEFAULT_SONNET_MODEL", model],
    ["ANTHROPIC_DEFAULT_HAIKU_MODEL", model],
  ];
  // Claude Code assumes a 1M window for a custom model and does NOT read the
  // server's max_model_len, so it overflows a smaller deployment. There is no
  // per-model context override; the only lever is disabling the 1M window,
  // which drops Claude Code to its standard ~200K. That fits a 256K deployment
  // (with headroom to spare). Only emit it for a sub-1M served context; a true
  // >=1M deployment keeps the 1M window. NOTE: Claude Code's floor is ~200K, so
  // a sub-200K deployment can still overflow — this can't fully fix those.
  if (maxModelLen != null && maxModelLen > 0 && maxModelLen < 1_000_000) {
    pairs.push(["CLAUDE_CODE_DISABLE_1M_CONTEXT", "1"]);
  }
  return pairs;
}

/** POSIX single-quote: wrap in '...', closing/escaping/reopening for any '. */
function bashQuote(v: string): string {
  return `'${v.split("'").join(`'\\''`)}'`;
}

/** PowerShell single-quote literal: double any embedded single quote. */
function pwshQuote(v: string): string {
  return `'${v.split("'").join("''")}'`;
}

/** Render the export block that sets up a shell to drive Claude Code against a
 *  vLLM deployment. Pure + deterministic. NOTE: baseUrl must be the server root
 *  with NO /v1 suffix — the Anthropic Messages API lives at the root. */
export function buildClaudeLaunchSnippet(input: {
  baseUrl: string;
  model: string;
  authToken: string;
  shell: LaunchShell;
  /** Served context window (from /v1/models). When < 1M, the snippet disables
   *  Claude Code's 1M window so it doesn't overflow the deployment. */
  maxModelLen?: number;
}): string {
  const { baseUrl, model, authToken, shell, maxModelLen } = input;
  const pairs = envPairs(baseUrl, model, authToken, maxModelLen);
  const lines =
    shell === "bash"
      ? pairs.map(([k, v]) => `export ${k}=${bashQuote(v)}`)
      : pairs.map(([k, v]) => `$env:${k} = ${pwshQuote(v)}`);
  lines.push("# then run: claude");
  return lines.join("\n");
}
