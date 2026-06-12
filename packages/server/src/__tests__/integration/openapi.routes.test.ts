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
  it("serves Swagger UI HTML", async () => {
    const res = await request(app()).get("/api/docs/");
    expect(res.status).toBe(200);
    expect(res.text).toMatch(/swagger-ui/i);
  });
});
