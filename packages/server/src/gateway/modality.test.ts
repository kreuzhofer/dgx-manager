import { describe, it, expect } from "vitest";
import { fc, it as itProp } from "@fast-check/vitest";
import { deploymentModality, partitionByModality, PATH_MODALITY } from "./modality.js";

describe("deploymentModality", () => {
  it("reads a recipe-declared image modality", () => {
    expect(deploymentModality(JSON.stringify({ runner: "dgxrun", modality: "image" }))).toBe("image");
  });

  /** Every deployment created before this field existed served text. A config
   *  without the key must keep meaning exactly that, or the whole existing
   *  fleet stops matching the chat path on the first deploy after upgrade. */
  it("defaults to text when the config omits modality", () => {
    expect(deploymentModality(JSON.stringify({ runner: "dgxrun" }))).toBe("text");
  });

  it("defaults to text for a null config", () => {
    expect(deploymentModality(null)).toBe("text");
  });

  /** The column is free-form text written by several code paths and by hand for
   *  inline-YAML deploys. Unparseable JSON must not take the gateway down. */
  it("defaults to text for a malformed config blob", () => {
    expect(deploymentModality("{not json")).toBe("text");
  });

  it("ignores an unrecognised modality rather than trusting it", () => {
    expect(deploymentModality(JSON.stringify({ modality: "video" }))).toBe("text");
  });
});

/** Whatever is in the config column, the answer is always one of the modalities
 *  the gateway knows how to route — never a value read straight out of the blob.
 *  This is the invariant that keeps an attacker-supplied or typo'd config from
 *  inventing a modality no path serves. */
itProp.prop([fc.anything()])("is total over any JSON-serialisable config", (v) => {
  let blob: string | null;
  try {
    blob = JSON.stringify({ modality: v });
  } catch {
    return true; // circular structures never reach the column
  }
  return (["text", "image"] as const).includes(deploymentModality(blob));
});

describe("partitionByModality", () => {
  const members = [
    { id: "a", m: "text" as const },
    { id: "b", m: "image" as const },
    { id: "c", m: "text" as const },
  ];

  it("keeps only members serving the required modality, preserving order", () => {
    const { matching, mismatched } = partitionByModality(members, (x) => x.m, "text");
    expect(matching.map((x) => x.id)).toEqual(["a", "c"]);
    expect(mismatched.map((x) => x.id)).toEqual(["b"]);
  });

  it("keeps only the image member when the image path asks", () => {
    const { matching, mismatched } = partitionByModality(members, (x) => x.m, "image");
    expect(matching.map((x) => x.id)).toEqual(["b"]);
    expect(mismatched.map((x) => x.id)).toEqual(["a", "c"]);
  });

  /** A pool whose every member is the wrong modality is the case worth naming
   *  in the refusal — the name resolved, the surface did not. This is what a
   *  client asking an all-text pool for an image hits. */
  it("reports an all-mismatched pool as empty with every member explained", () => {
    const textOnly = [
      { id: "a", m: "text" as const },
      { id: "c", m: "text" as const },
    ];
    const { matching, mismatched } = partitionByModality(textOnly, (x) => x.m, "image");
    expect(matching).toEqual([]);
    expect(mismatched.map((x) => x.id)).toEqual(["a", "c"]);
  });
});

describe("PATH_MODALITY", () => {
  /** The image path must never be satisfiable by a text deployment: that is the
   *  hang this whole module exists to prevent. */
  it("maps each forwarded path to the modality that can serve it", () => {
    expect(PATH_MODALITY["/v1/chat/completions"]).toBe("text");
    expect(PATH_MODALITY["/v1/embeddings"]).toBe("text");
    expect(PATH_MODALITY["/v1/images/generations"]).toBe("image");
  });
});
