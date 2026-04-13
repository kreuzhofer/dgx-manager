import { Router } from "express";
import { randomBytes } from "crypto";
import { prisma } from "../prisma.js";
export const tokensRouter = Router();

// POST /api/tokens — generate a join token
tokensRouter.post("/", async (req, res) => {
  const { label, expiresAt } = req.body || {};
  const token = randomBytes(32).toString("hex");

  const record = await prisma.joinToken.create({
    data: {
      token,
      label: label || null,
      expiresAt: expiresAt ? new Date(expiresAt) : null,
    },
  });

  res.json({ ...record, token }); // Return full token only on creation
});

// GET /api/tokens — list all tokens (masked)
tokensRouter.get("/", async (_req, res) => {
  const tokens = await prisma.joinToken.findMany({
    orderBy: { createdAt: "desc" },
  });

  const masked = tokens.map((t) => ({
    id: t.id,
    label: t.label,
    tokenSuffix: t.token.slice(-8),
    expiresAt: t.expiresAt,
    usedAt: t.usedAt,
    usedByNodeId: t.usedByNodeId,
    revokedAt: t.revokedAt,
    createdAt: t.createdAt,
    status: t.revokedAt
      ? "revoked"
      : t.usedAt
        ? "used"
        : t.expiresAt && t.expiresAt < new Date()
          ? "expired"
          : "active",
  }));

  res.json(masked);
});

// DELETE /api/tokens/:id — revoke a token
tokensRouter.delete("/:id", async (req, res) => {
  await prisma.joinToken.update({
    where: { id: req.params.id },
    data: { revokedAt: new Date() },
  });
  res.json({ status: "revoked" });
});
