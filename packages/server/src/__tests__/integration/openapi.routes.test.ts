import { describe, it, expect } from "vitest";
import request from "supertest";
import express from "express";
import { mountOpenApi } from "../../openapi.js";

function app() { const a = express(); mountOpenApi(a); return a; }

describe("OpenAPI", () => {
  it("serves a 3.x spec with system overview + domain tags", async () => {
    const res = await request(app()).get("/api/openapi.json");
    expect(res.status).toBe(200);
    expect(res.body.openapi).toMatch(/^3\./);
    expect(res.body.info.title).toMatch(/DGX Manager/i);
    expect(res.body.info.description.length).toBeGreaterThan(200);
    const tagNames = (res.body.tags ?? []).map((t: any) => t.name);
    expect(tagNames).toEqual(expect.arrayContaining(["Deployments", "Nodes", "Recipes"]));
  });
  // The load balancer was removed in favour of the inference gateway (ADR 0001).
  // It was a *published* part of the API contract, so its absence is asserted
  // rather than merely no longer checked — a stray re-mount would otherwise
  // silently restore a documented route nothing implements.
  it("no longer publishes the removed load balancer", async () => {
    const res = await request(app()).get("/api/openapi.json");
    const pathKeys = Object.keys(res.body.paths ?? {});
    expect(pathKeys.filter((p) => p.startsWith("/api/lb"))).toEqual([]);
    const tagNames = (res.body.tags ?? []).map((t: any) => t.name);
    expect(tagNames).not.toContain("Load Balancer");
    expect(res.body.info.description).not.toMatch(/load balancer/i);
  });
  it("serves Swagger UI HTML", async () => {
    const res = await request(app()).get("/api/docs/");
    expect(res.status).toBe(200);
    expect(res.text).toMatch(/swagger-ui/i);
  });
  it("documents key operations with comprehensive descriptions + tags", async () => {
    const res = await request(app()).get("/api/openapi.json");
    const paths = res.body.paths ?? {};
    const deploy = paths["/api/deployments"]?.post;
    expect(deploy).toBeTruthy();
    expect(deploy.tags).toContain("Deployments");
    expect((deploy.description ?? "").length).toBeGreaterThan(80);
    expect(paths["/api/recipes"]?.get?.tags).toContain("Recipes");
    expect(paths["/api/nodes"]?.get?.tags).toContain("Nodes");
    expect(JSON.stringify(deploy.requestBody ?? {})).toMatch(/recipeYaml/);
  });
  it("has non-empty paths covering all 12 routers", async () => {
    const res = await request(app()).get("/api/openapi.json");
    const paths = res.body.paths ?? {};
    const pathKeys = Object.keys(paths);
    // Must have substantial coverage
    expect(pathKeys.length).toBeGreaterThan(40);
    // Spot-check one path per router
    expect(paths["/api/nodes"]).toBeTruthy();
    expect(paths["/api/models"]).toBeTruthy();
    expect(paths["/api/deployments"]).toBeTruthy();
    expect(paths["/api/finetune"]).toBeTruthy();
    expect(paths["/api/recipes"]).toBeTruthy();
    expect(paths["/api/training-recipes"]).toBeTruthy();
    expect(paths["/api/tokens"]).toBeTruthy();
    expect(paths["/api/settings"]).toBeTruthy();
    expect(paths["/api/ollama-catalog/catalog"]).toBeTruthy();
    expect(paths["/api/agent/bundle"]).toBeTruthy();
    expect(paths["/api/datasets"]).toBeTruthy();
    expect(paths["/api/benchmarks"]).toBeTruthy();
    // Benchmarks POST must be annotated
    expect(paths["/api/benchmarks"]?.post).toBeTruthy();
    expect(paths["/api/benchmarks"]?.post?.tags).toContain("Benchmarks");
  });
});
