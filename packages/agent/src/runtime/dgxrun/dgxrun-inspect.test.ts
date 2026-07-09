import { describe, it, expect } from "vitest";
import { classifyDockerInspect } from "./dgxrun.js";

const NAME = "dgxrun_abc123";

describe("classifyDockerInspect", () => {
  it("reports found with state + restart count on a clean exit", () => {
    expect(classifyDockerInspect(0, "running 0\n", "", NAME)).toEqual({
      kind: "found", name: NAME, state: "running", restartCount: 0,
    });
    expect(classifyDockerInspect(0, "exited 3\n", "", NAME)).toEqual({
      kind: "found", name: NAME, state: "exited", restartCount: 3,
    });
  });

  it("treats a non-numeric restart count as 0 rather than NaN", () => {
    const r = classifyDockerInspect(0, "running x\n", "", NAME);
    expect(r).toMatchObject({ kind: "found", restartCount: 0 });
  });

  // Docker's positive "it does not exist" answer — the only case that may
  // legitimately tear down the cluster.
  it("reports absent when docker says No such object", () => {
    expect(classifyDockerInspect(1, "", `Error: No such object: ${NAME}`, NAME))
      .toEqual({ kind: "absent" });
  });

  it("reports absent for the 'No such container' phrasing too", () => {
    expect(classifyDockerInspect(1, "", "Error: No such container: x", NAME))
      .toEqual({ kind: "absent" });
  });

  // THE REGRESSION. A spawnSync timeout yields status===null + an error. This
  // used to collapse to null => "container missing" => the manager tore down all
  // four ranks of a healthy GLM-5.2 deploy that was mid-torch.compile.
  it("reports unknown (NOT absent) when docker inspect times out", () => {
    const r = classifyDockerInspect(null, "", "", NAME, new Error("spawnSync docker ETIMEDOUT"));
    expect(r.kind).toBe("unknown");
    expect(r).not.toEqual({ kind: "absent" });
  });

  it("reports unknown when the daemon errors for any other reason", () => {
    const r = classifyDockerInspect(1, "", "Cannot connect to the Docker daemon", NAME);
    expect(r.kind).toBe("unknown");
    if (r.kind === "unknown") expect(r.reason).toMatch(/daemon/i);
  });

  it("reports unknown on a zero exit with empty output", () => {
    const r = classifyDockerInspect(0, "   \n", "", NAME);
    expect(r.kind).toBe("unknown");
  });

  // Guard the classifier against a daemon error that merely *mentions* a
  // container: only docker's own "no such object/container" wording is absent.
  it("does not read an arbitrary error mentioning the name as absent", () => {
    const r = classifyDockerInspect(1, "", `permission denied while accessing ${NAME}`, NAME);
    expect(r.kind).toBe("unknown");
  });
});
