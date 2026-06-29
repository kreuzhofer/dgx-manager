import { prisma } from "../prisma.js";
import { registryRowsToWire, type RegistryWire } from "./wire.js";

export interface AgentSink {
  sendToAgent(nodeId: string, msg: Record<string, unknown>): void;
  getConnectedNodeIds(): string[];
}

// Exported (not destructured) so tests can spy on it via the module namespace.
export async function loadRegistryWire(): Promise<RegistryWire[]> {
  const rows = await prisma.sparkrunRegistry.findMany();
  return registryRowsToWire(rows);
}

export async function pushRegistriesToAgent(sink: AgentSink, nodeId: string): Promise<void> {
  const registries = await mod.loadRegistryWire();
  sink.sendToAgent(nodeId, { type: "cmd:set-registries", payload: { registries } });
}

export async function pushRegistriesToConnectedAgents(sink: AgentSink): Promise<void> {
  const registries = await mod.loadRegistryWire();
  for (const nodeId of sink.getConnectedNodeIds()) {
    sink.sendToAgent(nodeId, { type: "cmd:set-registries", payload: { registries } });
  }
}

import * as mod from "./push.js";
