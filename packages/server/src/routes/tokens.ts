import { Router } from "express";
import { randomBytes } from "crypto";
import { prisma } from "../prisma.js";
export const tokensRouter = Router();

/**
 * @openapi
 * /api/tokens:
 *   post:
 *     tags: [Tokens]
 *     summary: Generate a new node join token
 *     description: >
 *       Creates a cryptographically random 64-hex-character join token that an agent
 *       can use to self-register with the manager (GET /api/agent/install.sh embeds
 *       the token into the install command). The full token value is returned only
 *       on creation — subsequent GET requests return only the last 8 characters.
 *       Tokens can be scoped with an optional label and expiry date.
 *     requestBody:
 *       required: false
 *       content:
 *         application/json:
 *           schema:
 *             type: object
 *             properties:
 *               label: { type: string, description: "Optional human-readable description of who/what the token is for." }
 *               expiresAt: { type: string, format: date-time, description: "Optional ISO-8601 expiry timestamp." }
 *     responses:
 *       '200':
 *         description: Token record including the full `token` value (only returned once)
 */
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

/**
 * @openapi
 * /api/tokens:
 *   get:
 *     tags: [Tokens]
 *     summary: List all join tokens (masked)
 *     description: >
 *       Returns all JoinToken records ordered by creation date descending. The full
 *       token value is never returned here — only the last 8 characters
 *       (`tokenSuffix`) for identification. Each record includes a computed `status`
 *       field: `active`, `used`, `expired`, or `revoked`. Used by the dashboard
 *       to show which tokens have been consumed and whether any have been revoked.
 *     responses:
 *       '200':
 *         description: Array of masked token records with status computed field
 */
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

/**
 * @openapi
 * /api/tokens/{id}:
 *   delete:
 *     tags: [Tokens]
 *     summary: Revoke a join token
 *     description: >
 *       Sets `revokedAt` on the token to the current timestamp. Revoked tokens
 *       are rejected by the agent registration flow. This operation is irreversible
 *       via the API (the row is kept for audit purposes, not hard-deleted).
 *     parameters:
 *       - in: path
 *         name: id
 *         required: true
 *         schema: { type: string }
 *     responses:
 *       '200':
 *         description: '{ status: "revoked" }'
 */
tokensRouter.delete("/:id", async (req, res) => {
  await prisma.joinToken.update({
    where: { id: req.params.id },
    data: { revokedAt: new Date() },
  });
  res.json({ status: "revoked" });
});
