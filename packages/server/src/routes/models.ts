import { Router } from "express";
import { prisma } from "../prisma.js";

export const modelsRouter = Router();

modelsRouter.get("/", async (_req, res) => {
  const models = await prisma.model.findMany({
    orderBy: { createdAt: "desc" },
    include: { deployments: true },
  });
  res.json(models);
});

modelsRouter.post("/", async (req, res) => {
  const { name, runtime, parameters } = req.body;
  if (!name || !runtime) {
    return res.status(400).json({ error: "name and runtime required" });
  }
  const model = await prisma.model.create({
    data: { name, runtime, parameters },
  });
  res.status(201).json(model);
});

modelsRouter.delete("/:id", async (req, res) => {
  await prisma.model.delete({ where: { id: req.params.id } });
  res.json({ deleted: true });
});
