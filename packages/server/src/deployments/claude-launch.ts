export type LaunchShell = "bash" | "powershell";

/** Throwaway bearer token — vLLM ignores its value; its presence stops
 *  Claude Code from starting the interactive OAuth login flow. */
export const CLAUDE_AUTH_TOKEN = "dgx-local";

/** Ordered so the rendered snippet is deterministic. All three model-tier
 *  vars map to the one served model, so whichever tier Claude Code picks
 *  hits this deployment. */
function envPairs(baseUrl: string, model: string, authToken: string): [string, string][] {
  return [
    ["ANTHROPIC_BASE_URL", baseUrl],
    ["ANTHROPIC_AUTH_TOKEN", authToken],
    ["ANTHROPIC_DEFAULT_OPUS_MODEL", model],
    ["ANTHROPIC_DEFAULT_SONNET_MODEL", model],
    ["ANTHROPIC_DEFAULT_HAIKU_MODEL", model],
  ];
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
}): string {
  const { baseUrl, model, authToken, shell } = input;
  const pairs = envPairs(baseUrl, model, authToken);
  const lines =
    shell === "bash"
      ? pairs.map(([k, v]) => `export ${k}=${bashQuote(v)}`)
      : pairs.map(([k, v]) => `$env:${k} = ${pwshQuote(v)}`);
  lines.push("# then run: claude");
  return lines.join("\n");
}
