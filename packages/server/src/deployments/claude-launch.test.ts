import { describe, it, expect } from "vitest";
import { fc, it as fcIt } from "@fast-check/vitest";
import { buildClaudeLaunchSnippet, CLAUDE_AUTH_TOKEN } from "./claude-launch.js";

const VAR_NAMES = [
  "ANTHROPIC_BASE_URL",
  "ANTHROPIC_AUTH_TOKEN",
  "ANTHROPIC_DEFAULT_OPUS_MODEL",
  "ANTHROPIC_DEFAULT_SONNET_MODEL",
  "ANTHROPIC_DEFAULT_HAIKU_MODEL",
] as const;

// Reverse the POSIX single-quote escaping the builder applies, so the property
// verifies a genuine round-trip rather than re-deriving the encoder.
function bashUnquote(rhs: string): string {
  // rhs looks like: '...'  where inner ' were replaced by the 4 chars '\''
  return rhs.slice(1, -1).split(`'\\''`).join("'");
}
function pwshUnquote(rhs: string): string {
  // rhs looks like: '...'  where inner ' were doubled to ''
  return rhs.slice(1, -1).split("''").join("'");
}
// Pull the quoted RHS for a given var out of a rendered snippet.
function rhsOf(snippet: string, varName: string, shell: "bash" | "powershell"): string {
  const prefix = shell === "bash" ? `export ${varName}=` : `$env:${varName} = `;
  const line = snippet.split("\n").find((l) => l.startsWith(prefix));
  if (!line) throw new Error(`no line for ${varName}`);
  return line.slice(prefix.length);
}

describe("buildClaudeLaunchSnippet", () => {
  it("renders the bash export block with a trailing run hint", () => {
    const out = buildClaudeLaunchSnippet({
      baseUrl: "http://10.0.0.5:8000",
      model: "glm-5.2",
      authToken: "dgx-local",
      shell: "bash",
    });
    expect(out).toContain("export ANTHROPIC_BASE_URL='http://10.0.0.5:8000'");
    expect(out).toContain("export ANTHROPIC_DEFAULT_OPUS_MODEL='glm-5.2'");
    expect(out.trimEnd().endsWith("# then run: claude")).toBe(true);
    // Base URL must not carry a /v1 suffix.
    expect(out).not.toContain("/v1");
  });

  it("renders the PowerShell block with $env: assignments", () => {
    const out = buildClaudeLaunchSnippet({
      baseUrl: "http://10.0.0.5:8000",
      model: "glm-5.2",
      authToken: "dgx-local",
      shell: "powershell",
    });
    expect(out).toContain("$env:ANTHROPIC_BASE_URL = 'http://10.0.0.5:8000'");
    expect(out).toContain("$env:ANTHROPIC_DEFAULT_HAIKU_MODEL = 'glm-5.2'");
  });

  it("escapes embedded single quotes in both shells", () => {
    const model = "weird'name";
    const bash = buildClaudeLaunchSnippet({ baseUrl: "http://x:8000", model, authToken: "dgx-local", shell: "bash" });
    expect(bash).toContain(`export ANTHROPIC_DEFAULT_OPUS_MODEL='weird'\\''name'`);
    const pwsh = buildClaudeLaunchSnippet({ baseUrl: "http://x:8000", model, authToken: "dgx-local", shell: "powershell" });
    expect(pwsh).toContain(`$env:ANTHROPIC_DEFAULT_OPUS_MODEL = 'weird''name'`);
  });

  it("uses the exact throwaway auth token constant", () => {
    expect(CLAUDE_AUTH_TOKEN).toBe("dgx-local");
  });

  /**
   * Invariant: for any baseUrl / model / token, every ANTHROPIC_* var is present
   * and each value survives the round trip through the shell quoting unchanged.
   */
  const noNewline = fc.string().filter((s) => !/[\r\n]/.test(s));
  fcIt.prop([noNewline, noNewline, noNewline])(
    "round-trips all values through bash and powershell quoting",
    (baseUrl, model, authToken) => {
      for (const shell of ["bash", "powershell"] as const) {
        const out = buildClaudeLaunchSnippet({ baseUrl, model, authToken, shell });
        const unquote = shell === "bash" ? bashUnquote : pwshUnquote;
        for (const name of VAR_NAMES) {
          const recovered = unquote(rhsOf(out, name, shell));
          const expected =
            name === "ANTHROPIC_BASE_URL" ? baseUrl :
            name === "ANTHROPIC_AUTH_TOKEN" ? authToken : model;
          expect(recovered).toBe(expected);
        }
      }
    },
  );
});

describe("buildClaudeLaunchSnippet — 1M context cap", () => {
  const base = { baseUrl: "http://10.0.0.5:8000", model: "glm-5.2", authToken: "dgx-local", shell: "bash" as const };
  const DISABLE = "CLAUDE_CODE_DISABLE_1M_CONTEXT";

  it("adds CLAUDE_CODE_DISABLE_1M_CONTEXT for a sub-1M served context (256K)", () => {
    expect(buildClaudeLaunchSnippet({ ...base, maxModelLen: 262144 })).toContain(`export ${DISABLE}='1'`);
  });
  it("emits it in PowerShell too", () => {
    expect(buildClaudeLaunchSnippet({ ...base, shell: "powershell", maxModelLen: 262144 })).toContain(`$env:${DISABLE} = '1'`);
  });
  it("omits it when the served context is >= 1M (a real 1M model keeps 1M)", () => {
    expect(buildClaudeLaunchSnippet({ ...base, maxModelLen: 1_000_000 })).not.toContain(DISABLE);
  });
  it("omits it when max_model_len is unknown (endpoint unreachable)", () => {
    expect(buildClaudeLaunchSnippet({ ...base })).not.toContain(DISABLE);
  });
  it("still emits for a sub-200K deployment (best-effort; Claude Code's floor is ~200K so it may still overflow)", () => {
    expect(buildClaudeLaunchSnippet({ ...base, maxModelLen: 147456 })).toContain(`export ${DISABLE}='1'`);
  });
  it("omits it for a non-positive max_model_len", () => {
    expect(buildClaudeLaunchSnippet({ ...base, maxModelLen: 0 })).not.toContain(DISABLE);
  });
});
