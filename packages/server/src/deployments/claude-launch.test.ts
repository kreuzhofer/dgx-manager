import { describe, it, expect } from "vitest";
import { fc, it as fcIt } from "@fast-check/vitest";
import { buildClaudeLaunchSnippet } from "./claude-launch.js";

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

  /**
   * Invariant: for any baseUrl / model / token, every ANTHROPIC_* var is present
   * and each value survives the round trip through the shell quoting unchanged.
   */
  fcIt.prop([fc.string(), fc.string(), fc.string()])(
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
