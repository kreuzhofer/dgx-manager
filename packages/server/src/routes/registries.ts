import { Router } from "express";
import { prisma } from "../prisma.js";
import { validateRegistry } from "../registries/validate.js";
import { pushRegistriesToConnectedAgents, type AgentSink } from "../registries/push.js";

export const registriesRouter = Router();

/**
 * @openapi
 * /api/registries:
 *   get:
 *     tags: [Registries]
 *     summary: List sparkrun registries
 *   post:
 *     tags: [Registries]
 *     summary: Create a registry (pushes to all online nodes)
 */
registriesRouter.get("/", async (_req, res) => {
  const rows = await prisma.sparkrunRegistry.findMany({ orderBy: [{ sortOrder: "asc" }, { name: "asc" }] });
  res.json(rows);
});

registriesRouter.post("/", async (req, res) => {
  const v = validateRegistry(req.body);
  if (!v.ok) return res.status(400).json({ error: v.error });
  const existing = await prisma.sparkrunRegistry.findUnique({ where: { name: v.value.name } });
  if (existing) return res.status(409).json({ error: `Registry '${v.value.name}' already exists` });
  const row = await prisma.sparkrunRegistry.create({ data: v.value });
  await pushRegistriesToConnectedAgents(req.app.get("agentHub") as AgentSink);
  return res.status(201).json(row);
});

/**
 * @openapi
 * /api/registries/{id}:
 *   patch:
 *     tags: [Registries]
 *     summary: Update a registry (pushes to all online nodes)
 *   delete:
 *     tags: [Registries]
 *     summary: Delete a registry (pushes to all online nodes)
 */
registriesRouter.patch("/:id", async (req, res) => {
  const current = await prisma.sparkrunRegistry.findUnique({ where: { id: req.params.id } });
  if (!current) return res.status(404).json({ error: "Not found" });
  // Validate the merged record so partial updates can't produce an invalid row.
  const merged = { ...current, ...req.body };
  const v = validateRegistry(merged);
  if (!v.ok) return res.status(400).json({ error: v.error });
  if (v.value.name !== current.name) {
    const clash = await prisma.sparkrunRegistry.findUnique({ where: { name: v.value.name } });
    if (clash) return res.status(409).json({ error: `Registry '${v.value.name}' already exists` });
  }
  const row = await prisma.sparkrunRegistry.update({ where: { id: req.params.id }, data: v.value });
  await pushRegistriesToConnectedAgents(req.app.get("agentHub") as AgentSink);
  return res.json(row);
});

registriesRouter.delete("/:id", async (req, res) => {
  const existing = await prisma.sparkrunRegistry.findUnique({ where: { id: req.params.id } });
  if (existing) {
    const count = await prisma.sparkrunRegistry.count();
    if (count <= 1) {
      return res.status(409).json({ error: "Cannot delete the last registry — at least one must remain." });
    }
    await prisma.sparkrunRegistry.delete({ where: { id: req.params.id } });
  }
  await pushRegistriesToConnectedAgents(req.app.get("agentHub") as AgentSink);
  return res.json({ status: "deleted" });
});
