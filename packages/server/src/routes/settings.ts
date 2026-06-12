import { Router } from "express";
import { prisma } from "../prisma.js";
export const settingsRouter = Router();

/**
 * @openapi
 * /api/settings:
 *   get:
 *     tags: [Settings]
 *     summary: Return all server settings as a key-value map
 *     description: >
 *       Returns every Setting row as a flat JSON object (`{ [key]: value }`).
 *       Settings are persisted in SQLite and survive server restarts. Common
 *       keys include HuggingFace API tokens, NFS paths, and feature flags.
 *       Used by the dashboard Settings page to pre-populate form fields.
 *     responses:
 *       '200':
 *         description: Key-value map of all settings
 *         content:
 *           application/json:
 *             schema:
 *               type: object
 *               additionalProperties: { type: string }
 */
settingsRouter.get("/", async (_req, res) => {
  const rows = await prisma.setting.findMany();
  const settings: Record<string, string> = {};
  for (const row of rows) {
    settings[row.key] = row.value;
  }
  res.json(settings);
});

/**
 * @openapi
 * /api/settings:
 *   put:
 *     tags: [Settings]
 *     summary: Upsert server settings
 *     description: >
 *       Accepts a flat JSON object and upserts each key-value pair into the Settings
 *       table. Values are coerced to strings. Existing keys are updated; new keys are
 *       created. Partial updates are supported — keys not present in the request body
 *       are left unchanged.
 *     requestBody:
 *       required: true
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             additionalProperties: { type: string }
 *             description: Map of setting keys to their new string values
 *     responses:
 *       '200':
 *         description: '{ status: "ok" }'
 */
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
