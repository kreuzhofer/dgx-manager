import { Router } from "express";
import { prisma } from "../prisma.js";
export const settingsRouter = Router();

// GET /api/settings — return all settings as key-value object
settingsRouter.get("/", async (_req, res) => {
  const rows = await prisma.setting.findMany();
  const settings: Record<string, string> = {};
  for (const row of rows) {
    settings[row.key] = row.value;
  }
  res.json(settings);
});

// PUT /api/settings — upsert settings
settingsRouter.put("/", async (req, res) => {
  const entries = Object.entries(req.body || {}) as [string, string][];
  for (const [key, value] of entries) {
    await prisma.setting.upsert({
      where: { key },
      create: { key, value: String(value) },
      update: { value: String(value) },
    });
  }
  res.json({ status: "ok" });
});
