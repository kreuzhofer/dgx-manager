# Node Power Management (Reboot / Shutdown / Wake-on-LAN) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add per-node Reboot and Shutdown buttons (with confirmation) to each node card, keep powered-off nodes visible as inactive cards, and add a Wake-on-LAN button to bring them back — with a conditional contingency phase if WOL does not work on this hardware.

**Architecture:** Server gains two REST endpoints on the existing nodes router (`POST /api/nodes/:id/power` and `POST /api/nodes/:id/wake`). Power actions run over the existing `sshExec` primitive using systemd's `--no-block` so the SSH call returns cleanly before the node drops. A new `Node.powerState` column tracks deliberate power intent (`on` | `rebooting` | `off` | `waking` | `asleep`) separately from the agent-connection `status` field. WOL uses a dependency-free UDP magic packet sent from the manager; the node MAC is captured server-side over SSH (no agent change). The dashboard adds confirm-gated buttons to `node-card.tsx` and renders an inactive style when `powerState === "off"`.

**Tech Stack:** Express 5 + Prisma (SQLite) on the server; `ssh2` (already used via `sshExec`); Node `dgram` for the WOL magic packet (no new deps); Next.js 15 / React 19 dashboard; Vitest + fast-check + supertest for tests.

**Scope note (read before starting):** This plan spans three subsystems and is organized into three phases that each produce working, testable software on their own:
- **Phase 1 — Reboot / Shutdown + inactive cards** (self-contained, shippable alone).
- **Phase 2 — Wake-on-LAN** (builds on Phase 1; ends with a manual hardware test round + a decision gate).
- **Phase 3 — CONTINGENCY: Suspend + wake-from-sleep** (execute ONLY if the Phase 2 WOL test round fails). If WOL works, skip Phase 3 entirely.

**Key risks the test round must resolve (do not assume these work):**
1. **Passwordless sudo** for `systemctl poweroff/reboot` must already be configured for `$SSH_USER` on each node (the provisioner already runs `sudo` for installs, so this is expected — but the endpoint must surface a clear error if sudo prompts for a password).
2. **WOL broadcast reachability from inside the manager Docker container** is unknown — a container on a bridge network may not deliver an L2 broadcast to the cluster subnet. This is the single biggest WOL unknown and is exactly what the Phase 2 test round validates.
3. **DGX Spark / GB10 NIC + BIOS WOL support from full power-off (S5)** is unverified — hence the Phase 3 contingency (wake from suspend/S3 instead).

**Conventions to follow (already in this repo):**
- Mandatory agent version bump applies ONLY when editing `packages/agent/src/*`. This plan does **not** touch the agent, so no bump is required.
- Schema changes are applied with `npm run db:push` (no migrations directory).
- `npm test` must be green before claiming any task done.

---

## File Structure

**Server (new):**
- `packages/server/src/nodes/power.ts` — pure helpers: `powerCommand(action)`, `macCaptureCmd(ip)`, `normalizeMac(raw)`. No IO.
- `packages/server/src/nodes/wol.ts` — `isValidMac`, `buildMagicPacket(mac)`, `broadcastFor(ip, prefix)` (pure) + `sendMagicPacket(mac, broadcast, opts?)` (dgram IO).
- `packages/server/src/nodes/power.test.ts` — unit/property tests for `power.ts`.
- `packages/server/src/nodes/wol.test.ts` — unit/property tests for `wol.ts` pure helpers.
- `packages/server/src/__tests__/integration/nodes.power.test.ts` — supertest for `POST /:id/power`.
- `packages/server/src/__tests__/integration/nodes.wake.test.ts` — supertest for `POST /:id/wake`.

**Server (modified):**
- `prisma/schema.prisma` — add `powerState` and `macAddress` to `model Node`.
- `packages/server/src/routes/nodes.ts` — add `POST /:id/power` and `POST /:id/wake`.
- `packages/server/src/index.ts` — wire injectable defaults (`app.set("sshExec", ...)`, `app.set("wolSend", ...)`).
- `packages/server/src/ws/agent-hub.ts` — set `powerState: "on"` when an agent connects.
- `packages/server/src/ssh/provisioner.ts` — capture MAC during `auditNode` (best-effort).

**Dashboard (modified):**
- `packages/dashboard/components/node-card.tsx` — power buttons + confirm + inactive rendering.
- `packages/dashboard/app/page.tsx` — extend `Node` type + `node:status` SSE handler to carry `powerState`.

---

## Phase 1 — Reboot / Shutdown + Inactive Cards

### Task 1: Schema — add `powerState` and `macAddress` to Node

**Files:**
- Modify: `prisma/schema.prisma` (the `model Node { ... }` block)

- [ ] **Step 1: Add the two fields**

In `prisma/schema.prisma`, inside `model Node`, add these lines right after the `arch String?` line:

```prisma
  // Deliberate power intent, distinct from agent-connection `status`.
  // Values: "on" | "rebooting" | "off" | "waking" | "asleep".
  powerState      String           @default("on")
  // MAC of the interface bound to ipAddress, captured over SSH while the node
  // is reachable. Required for Wake-on-LAN. Null until first captured.
  macAddress      String?
```

- [ ] **Step 2: Apply the schema and regenerate the client**

Run: `npm run db:push`
Expected: Prisma prints "Your database is now in sync with your Prisma schema." and regenerates the client (no errors).

- [ ] **Step 3: Commit**

```bash
git add prisma/schema.prisma
git commit -m "feat(schema): add Node.powerState and Node.macAddress for power management"
```

---

### Task 2: Pure power helpers (`power.ts`)

**Files:**
- Create: `packages/server/src/nodes/power.ts`
- Test: `packages/server/src/nodes/power.test.ts`

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/nodes/power.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { powerCommand, macCaptureCmd, normalizeMac } from "./power.js";

describe("powerCommand", () => {
  // Uses systemd --no-block so the SSH exec returns BEFORE the node drops,
  // giving the caller a clean exit code instead of a dropped-connection error.
  it("maps reboot to a non-blocking systemd reboot", () => {
    expect(powerCommand("reboot")).toBe("sudo systemctl --no-block reboot");
  });
  it("maps shutdown to a non-blocking systemd poweroff", () => {
    expect(powerCommand("shutdown")).toBe("sudo systemctl --no-block poweroff");
  });
  it("maps sleep to systemd suspend", () => {
    expect(powerCommand("sleep")).toBe("sudo systemctl suspend");
  });
  it("throws on an unknown action", () => {
    // @ts-expect-error invalid action
    expect(() => powerCommand("explode")).toThrow();
  });
});

describe("macCaptureCmd", () => {
  it("builds an ssh command that finds the iface for an IP then reads its MAC", () => {
    const cmd = macCaptureCmd("192.168.44.41");
    expect(cmd).toContain("192.168.44.41");
    expect(cmd).toContain("/sys/class/net/");
  });
});

describe("normalizeMac", () => {
  it("lowercases and trims a captured MAC", () => {
    expect(normalizeMac("  AA:BB:CC:DD:EE:FF\n")).toBe("aa:bb:cc:dd:ee:ff");
  });
  it("returns null for empty or junk output", () => {
    expect(normalizeMac("")).toBeNull();
    expect(normalizeMac("device not found")).toBeNull();
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run packages/server/src/nodes/power.test.ts`
Expected: FAIL — "Failed to resolve import './power.js'".

- [ ] **Step 3: Write the implementation**

Create `packages/server/src/nodes/power.ts`:

```ts
export type PowerAction = "reboot" | "shutdown" | "sleep";

const COMMANDS: Record<PowerAction, string> = {
  // --no-block returns immediately without waiting for the systemd job, so the
  // SSH exec gets a clean exit code before the machine actually goes down.
  reboot: "sudo systemctl --no-block reboot",
  shutdown: "sudo systemctl --no-block poweroff",
  // suspend is interactive-safe and returns once the node has entered S3.
  sleep: "sudo systemctl suspend",
};

export function powerCommand(action: PowerAction): string {
  const cmd = COMMANDS[action];
  if (!cmd) throw new Error(`Unknown power action: ${action}`);
  return cmd;
}

/**
 * Shell command (run over SSH) that resolves the network interface whose IPv4
 * address matches `ip`, then prints that interface's MAC. Output is a single
 * MAC line (or empty if not found).
 */
export function macCaptureCmd(ip: string): string {
  return (
    `ifc=$(ip -o -4 addr show | awk -v ip="${ip}" '$4 ~ "^"ip"/" {print $2; exit}'); ` +
    `cat /sys/class/net/"$ifc"/address 2>/dev/null`
  );
}

const MAC_RE = /^([0-9a-f]{2}:){5}[0-9a-f]{2}$/;

/** Trim+lowercase a captured MAC, or null if it is not a valid MAC. */
export function normalizeMac(raw: string): string | null {
  const m = raw.trim().toLowerCase();
  return MAC_RE.test(m) ? m : null;
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run packages/server/src/nodes/power.test.ts`
Expected: PASS (all cases green).

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/nodes/power.ts packages/server/src/nodes/power.test.ts
git commit -m "feat(server): pure power-command + MAC-capture helpers"
```

---

### Task 3: `POST /api/nodes/:id/power` endpoint

**Files:**
- Modify: `packages/server/src/routes/nodes.ts` (add a new route; it already imports `prisma`, `AgentHub`, and broadcasts via `sse`)
- Modify: `packages/server/src/index.ts` (inject the real `sshExec` default)
- Test: `packages/server/src/__tests__/integration/nodes.power.test.ts`

- [ ] **Step 1: Write the failing integration test**

Create `packages/server/src/__tests__/integration/nodes.power.test.ts`. The per-suite SQLite + consent setup mirrors `deployments.vram-admission.test.ts`:

```ts
import { mkdtempSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { execSync } from "child_process";
import { beforeAll, afterEach, describe, it, expect, vi } from "vitest";

// DATABASE_URL must be set BEFORE importing prisma.
const dbFile = join(mkdtempSync(join(tmpdir(), "nodes-power-")), "test.db");
process.env.DATABASE_URL = `file:${dbFile}`;
process.env.PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION =
  "I understand this resets a per-test database";

import express from "express";
import request from "supertest";
import { prisma } from "../../db.js";
import { nodesRouter } from "../../routes/nodes.js";

beforeAll(() => {
  execSync("npx prisma db push --force-reset --skip-generate", {
    env: process.env,
    stdio: "ignore",
  });
});

afterEach(async () => {
  await prisma.metricSnapshot.deleteMany();
  await prisma.node.deleteMany();
});

function makeApp(sshExec: any) {
  const app = express();
  app.use(express.json());
  app.set("agentHub", { isAgentOnline: () => true, sendToAgent: () => {} });
  app.set("sshExec", sshExec);
  app.use("/api/nodes", nodesRouter);
  return app;
}

describe("POST /api/nodes/:id/power", () => {
  it("reboot: runs the systemd reboot command and sets powerState=rebooting", async () => {
    const node = await prisma.node.create({
      data: { name: "spark-test", ipAddress: "192.168.44.41" },
    });
    const sshExec = vi.fn().mockResolvedValue({ code: 0, stdout: "", stderr: "" });
    const app = makeApp(sshExec);

    const res = await request(app)
      .post(`/api/nodes/${node.id}/power`)
      .send({ action: "reboot" });

    expect(res.status).toBe(200);
    expect(res.body.powerState).toBe("rebooting");
    // last sshExec call is the reboot command against the node IP
    const lastCall = sshExec.mock.calls.at(-1);
    expect(lastCall[0]).toBe("192.168.44.41");
    expect(lastCall[1]).toBe("sudo systemctl --no-block reboot");
    const after = await prisma.node.findUnique({ where: { id: node.id } });
    expect(after?.powerState).toBe("rebooting");
  });

  it("shutdown: sets powerState=off and captures MAC first", async () => {
    const node = await prisma.node.create({
      data: { name: "spark-test", ipAddress: "192.168.44.41" },
    });
    // First call = MAC capture, second = poweroff.
    const sshExec = vi
      .fn()
      .mockResolvedValueOnce({ code: 0, stdout: "AA:BB:CC:DD:EE:FF", stderr: "" })
      .mockResolvedValueOnce({ code: 0, stdout: "", stderr: "" });
    const app = makeApp(sshExec);

    const res = await request(app)
      .post(`/api/nodes/${node.id}/power`)
      .send({ action: "shutdown" });

    expect(res.status).toBe(200);
    expect(res.body.powerState).toBe("off");
    const after = await prisma.node.findUnique({ where: { id: node.id } });
    expect(after?.powerState).toBe("off");
    expect(after?.macAddress).toBe("aa:bb:cc:dd:ee:ff");
  });

  it("returns 404 for an unknown node", async () => {
    const app = makeApp(vi.fn());
    const res = await request(app).post(`/api/nodes/nope/power`).send({ action: "reboot" });
    expect(res.status).toBe(404);
  });

  it("returns 400 for an invalid action", async () => {
    const node = await prisma.node.create({
      data: { name: "spark-test", ipAddress: "192.168.44.41" },
    });
    const app = makeApp(vi.fn());
    const res = await request(app).post(`/api/nodes/${node.id}/power`).send({ action: "explode" });
    expect(res.status).toBe(400);
  });

  it("returns 502 when the SSH command fails", async () => {
    const node = await prisma.node.create({
      data: { name: "spark-test", ipAddress: "192.168.44.41" },
    });
    const sshExec = vi.fn().mockRejectedValue(new Error("permission denied (sudo password?)"));
    const app = makeApp(sshExec);
    const res = await request(app).post(`/api/nodes/${node.id}/power`).send({ action: "reboot" });
    expect(res.status).toBe(502);
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run packages/server/src/__tests__/integration/nodes.power.test.ts`
Expected: FAIL — route returns 404 for every POST (the `/power` route does not exist yet).

- [ ] **Step 3: Implement the route**

In `packages/server/src/routes/nodes.ts`, add these imports near the existing imports at the top:

```ts
import { powerCommand, macCaptureCmd, normalizeMac, type PowerAction } from "../nodes/power.js";
import { broadcast as sseBroadcast } from "../sse.js";
import { sshExec as defaultSshExec } from "../ssh/executor.js";
```

Then add this route (place it after the `update-agent` route, before the `DELETE /:id` route):

```ts
// POST /api/nodes/:id/power — reboot / shutdown / sleep a node over SSH.
// powerState transitions: reboot -> "rebooting", shutdown -> "off", sleep -> "asleep".
// MAC is captured (best-effort) before a shutdown/sleep so the node can be woken later.
nodesRouter.post("/:id/power", async (req, res) => {
  const action = req.body?.action as PowerAction;
  if (action !== "reboot" && action !== "shutdown" && action !== "sleep") {
    return res.status(400).json({ error: `Invalid action: ${action}` });
  }
  const node = await prisma.node.findUnique({ where: { id: req.params.id } });
  if (!node) return res.status(404).json({ error: "Node not found" });
  if (!node.ipAddress) return res.status(400).json({ error: "Node has no ipAddress" });

  // Injected so tests can stub it; defaults to the real ssh2-backed exec.
  const sshExec = (req.app.get("sshExec") || defaultSshExec) as typeof defaultSshExec;

  // Best-effort MAC capture while the node is still reachable (skip for reboot —
  // it comes right back; capture for shutdown/sleep where we may need WOL).
  let macAddress = node.macAddress;
  if (action !== "reboot") {
    try {
      const r = await sshExec(node.ipAddress, macCaptureCmd(node.ipAddress), { timeout: 10_000 });
      const mac = normalizeMac(r.stdout);
      if (mac) macAddress = mac;
    } catch {
      // non-fatal: WOL just won't be available if we never captured a MAC
    }
  }

  try {
    await sshExec(node.ipAddress, powerCommand(action), { timeout: 15_000 });
  } catch (err) {
    return res.status(502).json({ error: `Power command failed: ${String(err)}` });
  }

  const powerState = action === "reboot" ? "rebooting" : action === "sleep" ? "asleep" : "off";
  await prisma.node.update({
    where: { id: node.id },
    data: { powerState, ...(macAddress ? { macAddress } : {}) },
  });
  sseBroadcast({ type: "node:status", payload: { nodeId: node.id, powerState } });

  res.json({ status: "ok", powerState });
});
```

> Note: if `sseBroadcast` is already imported in `nodes.ts`, do not add the duplicate import — reuse the existing one. Same for any existing `sshExec` import.

- [ ] **Step 4: Wire the injectable default in index.ts**

In `packages/server/src/index.ts`, near the existing `app.set("agentHub", ...)` line, add:

```ts
import { sshExec } from "./ssh/executor.js";
// ...
app.set("sshExec", sshExec);
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `npx vitest run packages/server/src/__tests__/integration/nodes.power.test.ts`
Expected: PASS (5/5).

- [ ] **Step 6: Commit**

```bash
git add packages/server/src/routes/nodes.ts packages/server/src/index.ts packages/server/src/__tests__/integration/nodes.power.test.ts
git commit -m "feat(server): POST /api/nodes/:id/power (reboot/shutdown/sleep over SSH)"
```

---

### Task 4: Clear `powerState` to "on" when an agent reconnects

**Files:**
- Modify: `packages/server/src/ws/agent-hub.ts` (the two `status: "online"` update sites, ~lines 178 and 245)

- [ ] **Step 1: Add `powerState: "on"` to the online updates**

In `packages/server/src/ws/agent-hub.ts`, find each `prisma.node.update` (or `upsert`) that sets `status: "online"` (there are two, near lines 178 and 245). Add `powerState: "on"` to the same `data` object, e.g.:

```ts
data: {
  status: "online",
  // a reconnecting agent means the box is back up — clear any off/rebooting/waking intent
  powerState: "on",
  // ...existing fields (agentVersion, lastSeen, etc.)
},
```

Do **not** modify the `ws.on("close")` handler — a deliberate "off" must survive the disconnect so the card stays inactive and WOL-eligible.

- [ ] **Step 2: Verify the build still typechecks**

Run: `npm run build -w packages/server` (or `npx tsc -p packages/server --noEmit`)
Expected: no type errors.

- [ ] **Step 3: Commit**

```bash
git add packages/server/src/ws/agent-hub.ts
git commit -m "feat(server): reset Node.powerState to on when an agent reconnects"
```

---

### Task 5: Capture MAC during node audit (best-effort)

**Files:**
- Modify: `packages/server/src/ssh/provisioner.ts` (inside `auditNode`, which already SSHes to the node)

- [ ] **Step 1: Capture + persist the MAC in `auditNode`**

In `packages/server/src/ssh/provisioner.ts`, add the import near the top:

```ts
import { macCaptureCmd, normalizeMac } from "../nodes/power.js";
import { prisma } from "../db.js";
```

Inside `auditNode(host, nodeId?)`, after the existing prereq checks but before it returns the report, add (only when `nodeId` is provided):

```ts
if (nodeId) {
  try {
    const r = await sshExec(host, macCaptureCmd(host), { timeout: 10_000 });
    const mac = normalizeMac(r.stdout);
    if (mac) await prisma.node.update({ where: { id: nodeId }, data: { macAddress: mac } });
  } catch {
    // non-fatal — audit should not fail because MAC capture failed
  }
}
```

> If `prisma` is already imported in `provisioner.ts`, reuse it. `host` here is the management IP, so `macCaptureCmd(host)` resolves the right interface.

- [ ] **Step 2: Verify the build typechecks**

Run: `npx tsc -p packages/server --noEmit`
Expected: no type errors.

- [ ] **Step 3: Commit**

```bash
git add packages/server/src/ssh/provisioner.ts
git commit -m "feat(server): capture node MAC during audit for future WOL"
```

---

### Task 6: Power buttons + inactive rendering on the node card

**Files:**
- Modify: `packages/dashboard/components/node-card.tsx`
- Modify: `packages/dashboard/app/page.tsx` (extend `Node` type + `node:status` SSE handler)

- [ ] **Step 1: Extend the Node types to carry powerState**

In `packages/dashboard/components/node-card.tsx`, add `powerState?: string;` to the `interface Node { ... }` block (right after `status: string;`).

In `packages/dashboard/app/page.tsx`, find the `Node` interface and add `powerState?: string;` to it as well.

- [ ] **Step 2: Carry powerState through the SSE handler**

In `packages/dashboard/app/page.tsx`, the `node:status` handler currently reads `{ nodeId, status }`. Replace it with a version that merges `powerState` when present (the power endpoint broadcasts `{ nodeId, powerState }` with no `status`):

```ts
if (event.type === "node:status") {
  const { nodeId, status, powerState } = event.payload as {
    nodeId: string;
    status?: string;
    powerState?: string;
  };
  setNodes((prev) =>
    prev.map((n) =>
      n.id === nodeId
        ? { ...n, ...(status ? { status } : {}), ...(powerState ? { powerState } : {}) }
        : n
    )
  );
}
```

- [ ] **Step 3: Add the power-control buttons + confirm + inactive style**

In `packages/dashboard/components/node-card.tsx`:

(a) Add a helper inside the `NodeCard` component body (before the `return`):

```ts
const isOff = node.powerState === "off" || node.powerState === "asleep";

async function power(action: "reboot" | "shutdown") {
  const verb = action === "reboot" ? "Reboot" : "Shut down";
  if (!window.confirm(`${verb} node "${node.name}"? This runs sudo on the machine and will drop its agent.`)) {
    return;
  }
  try {
    await apiFetch(`/api/nodes/${node.id}/power`, {
      method: "POST",
      body: JSON.stringify({ action }),
    });
  } catch (e) {
    window.alert(`Power action failed: ${e instanceof Error ? e.message : String(e)}`);
  }
}

async function wake() {
  try {
    await apiFetch(`/api/nodes/${node.id}/wake`, { method: "POST" });
  } catch (e) {
    window.alert(`Wake failed: ${e instanceof Error ? e.message : String(e)}`);
  }
}
```

(b) In the header's right-hand `<div className="flex items-center gap-2">` (the one with the status dot), add the buttons before the status dot:

```tsx
<div className="flex items-center gap-1">
  {isOff ? (
    <button
      onClick={wake}
      className="text-[10px] px-2 py-0.5 rounded bg-blue-900/60 text-blue-300 hover:bg-blue-800"
      title="Wake-on-LAN"
    >
      Wake
    </button>
  ) : (
    <>
      <button
        onClick={() => power("reboot")}
        className="text-[10px] px-2 py-0.5 rounded bg-gray-800 text-gray-300 hover:bg-gray-700"
      >
        Reboot
      </button>
      <button
        onClick={() => power("shutdown")}
        className="text-[10px] px-2 py-0.5 rounded bg-red-900/60 text-red-300 hover:bg-red-800"
      >
        Shutdown
      </button>
    </>
  )}
</div>
```

(c) Make the whole card dim when powered off. Change the outermost `<div className="bg-gray-900 border border-gray-800 ...">` to append an opacity class when `isOff`:

```tsx
<div className={`bg-gray-900 border border-gray-800 rounded-lg p-4 hover:border-gray-700 transition-colors ${isOff ? "opacity-60" : ""}`}>
```

(d) When `isOff`, show a "Powered off" label instead of "No metrics yet". Find the `) : (` branch that renders `<p className="text-xs text-gray-500 italic">No metrics yet</p>` and replace that `<p>` with:

```tsx
<p className="text-xs text-gray-500 italic">
  {isOff ? "Powered off — Wake to bring it back" : "No metrics yet"}
</p>
```

- [ ] **Step 4: Verify the dashboard builds**

Run: `npm run build -w packages/dashboard` (or `npx next build` in `packages/dashboard`)
Expected: build succeeds, no TypeScript errors.

- [ ] **Step 5: Commit**

```bash
git add packages/dashboard/components/node-card.tsx packages/dashboard/app/page.tsx
git commit -m "feat(dashboard): reboot/shutdown buttons + inactive powered-off card"
```

---

### Task 7: Phase 1 full test run

- [ ] **Step 1: Run the whole server test suite**

Run: `npm test`
Expected: all tests green (including the new `power.test.ts` and `nodes.power.test.ts`).

- [ ] **Step 2: Manual smoke (real node, optional but recommended before Phase 2)**

Bring up the app per CLAUDE.md (`./scripts/build-agent-bundles.sh && MANAGER_ADVERTISE_HOST=<ip> SSH_USER=<user> docker compose up -d --build`). On the overview page:
- Click **Reboot** on a test node, confirm → the node's agent disconnects then reconnects within ~1–2 min, card returns to online.
- Click **Shutdown** on a test node, confirm → card goes dim, shows "Powered off", `powerState` is `off` in the DB. (You will need physical/IPMI access or Phase 2 WOL to bring it back — pick a node you can recover.)

Record the result. If reboot/shutdown work, proceed to Phase 2.

---

## Phase 2 — Wake-on-LAN

### Task 8: Pure WOL helpers (`wol.ts`)

**Files:**
- Create: `packages/server/src/nodes/wol.ts`
- Test: `packages/server/src/nodes/wol.test.ts`

- [ ] **Step 1: Write the failing test**

Create `packages/server/src/nodes/wol.test.ts`:

```ts
import { describe, it, expect } from "vitest";
import { it as fcit } from "@fast-check/vitest";
import * as fc from "fast-check";
import { isValidMac, buildMagicPacket, broadcastFor } from "./wol.js";

describe("isValidMac", () => {
  it("accepts lowercase colon-separated MACs", () => {
    expect(isValidMac("aa:bb:cc:dd:ee:ff")).toBe(true);
  });
  it("rejects junk", () => {
    expect(isValidMac("nope")).toBe(false);
    expect(isValidMac("aa:bb:cc:dd:ee")).toBe(false);
  });
});

describe("broadcastFor", () => {
  it("computes the /24 broadcast by default", () => {
    expect(broadcastFor("192.168.44.41")).toBe("192.168.44.255");
  });
  it("supports an explicit /16 prefix", () => {
    expect(broadcastFor("192.168.44.41", 16)).toBe("192.168.255.255");
  });
});

// PROPERTY: a WOL magic packet is always exactly 102 bytes — a 6-byte 0xFF
// sync stream followed by the 6-byte MAC repeated 16 times.
fcit.prop([
  fc.array(fc.integer({ min: 0, max: 255 }), { minLength: 6, maxLength: 6 }),
])("buildMagicPacket is 6x0xFF + 16x MAC", (octets) => {
  const mac = octets.map((o) => o.toString(16).padStart(2, "0")).join(":");
  const pkt = buildMagicPacket(mac);
  expect(pkt.length).toBe(102);
  for (let i = 0; i < 6; i++) expect(pkt[i]).toBe(0xff);
  for (let rep = 0; rep < 16; rep++) {
    for (let b = 0; b < 6; b++) {
      expect(pkt[6 + rep * 6 + b]).toBe(octets[b]);
    }
  }
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run packages/server/src/nodes/wol.test.ts`
Expected: FAIL — "Failed to resolve import './wol.js'".

- [ ] **Step 3: Write the implementation**

Create `packages/server/src/nodes/wol.ts`:

```ts
import { createSocket } from "dgram";

const MAC_RE = /^([0-9a-f]{2}:){5}[0-9a-f]{2}$/;

export function isValidMac(mac: string): boolean {
  return MAC_RE.test(mac.trim().toLowerCase());
}

/** 102-byte magic packet: 6x 0xFF followed by the 6-byte MAC repeated 16x. */
export function buildMagicPacket(mac: string): Buffer {
  const m = mac.trim().toLowerCase();
  if (!MAC_RE.test(m)) throw new Error(`Invalid MAC: ${mac}`);
  const macBytes = Buffer.from(m.split(":").map((h) => parseInt(h, 16)));
  const packet = Buffer.alloc(102, 0xff);
  for (let i = 0; i < 16; i++) macBytes.copy(packet, 6 + i * 6);
  return packet;
}

/** Directed broadcast address for an IPv4 `ip` at CIDR `prefix` (default /24). */
export function broadcastFor(ip: string, prefix = 24): string {
  const ipNum = ip.split(".").reduce((acc, o) => (acc << 8) + (parseInt(o, 10) & 0xff), 0) >>> 0;
  const mask = prefix === 0 ? 0 : (0xffffffff << (32 - prefix)) >>> 0;
  const bcast = (ipNum | (~mask >>> 0)) >>> 0;
  return [bcast >>> 24, (bcast >>> 16) & 0xff, (bcast >>> 8) & 0xff, bcast & 0xff].join(".");
}

/**
 * Send the WOL magic packet. Sends to BOTH the directed subnet broadcast and
 * the global 255.255.255.255, on UDP ports 9 and 7, to maximize the chance one
 * path reaches the target's L2 segment. Resolves once all sends complete.
 */
export async function sendMagicPacket(
  mac: string,
  broadcast: string,
  opts?: { ports?: number[] },
): Promise<void> {
  const packet = buildMagicPacket(mac);
  const ports = opts?.ports ?? [9, 7];
  const targets = [broadcast, "255.255.255.255"];
  const sock = createSocket("udp4");
  await new Promise<void>((resolve, reject) => {
    sock.bind(() => {
      sock.setBroadcast(true);
      let pending = targets.length * ports.length;
      let failed: Error | null = null;
      for (const t of targets) {
        for (const p of ports) {
          sock.send(packet, p, t, (err) => {
            if (err) failed = err;
            if (--pending === 0) {
              sock.close();
              failed ? reject(failed) : resolve();
            }
          });
        }
      }
    });
  });
}
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `npx vitest run packages/server/src/nodes/wol.test.ts`
Expected: PASS (property + unit cases green).

- [ ] **Step 5: Commit**

```bash
git add packages/server/src/nodes/wol.ts packages/server/src/nodes/wol.test.ts
git commit -m "feat(server): Wake-on-LAN magic-packet + broadcast helpers"
```

---

### Task 9: `POST /api/nodes/:id/wake` endpoint

**Files:**
- Modify: `packages/server/src/routes/nodes.ts`
- Modify: `packages/server/src/index.ts` (inject the real `sendMagicPacket` default)
- Test: `packages/server/src/__tests__/integration/nodes.wake.test.ts`

- [ ] **Step 1: Write the failing integration test**

Create `packages/server/src/__tests__/integration/nodes.wake.test.ts` (same per-suite DB setup as Task 3):

```ts
import { mkdtempSync } from "fs";
import { tmpdir } from "os";
import { join } from "path";
import { execSync } from "child_process";
import { beforeAll, afterEach, describe, it, expect, vi } from "vitest";

const dbFile = join(mkdtempSync(join(tmpdir(), "nodes-wake-")), "test.db");
process.env.DATABASE_URL = `file:${dbFile}`;
process.env.PRISMA_USER_CONSENT_FOR_DANGEROUS_AI_ACTION =
  "I understand this resets a per-test database";

import express from "express";
import request from "supertest";
import { prisma } from "../../db.js";
import { nodesRouter } from "../../routes/nodes.js";

beforeAll(() => {
  execSync("npx prisma db push --force-reset --skip-generate", {
    env: process.env,
    stdio: "ignore",
  });
});
afterEach(async () => {
  await prisma.node.deleteMany();
});

function makeApp(wolSend: any) {
  const app = express();
  app.use(express.json());
  app.set("agentHub", { isAgentOnline: () => false, sendToAgent: () => {} });
  app.set("wolSend", wolSend);
  app.use("/api/nodes", nodesRouter);
  return app;
}

describe("POST /api/nodes/:id/wake", () => {
  it("sends a magic packet to the node MAC + /24 broadcast and sets powerState=waking", async () => {
    const node = await prisma.node.create({
      data: { name: "spark-test", ipAddress: "192.168.44.41", macAddress: "aa:bb:cc:dd:ee:ff", powerState: "off" },
    });
    const wolSend = vi.fn().mockResolvedValue(undefined);
    const res = await request(makeApp(wolSend)).post(`/api/nodes/${node.id}/wake`).send();
    expect(res.status).toBe(200);
    expect(res.body.powerState).toBe("waking");
    expect(wolSend).toHaveBeenCalledWith("aa:bb:cc:dd:ee:ff", "192.168.44.255");
    const after = await prisma.node.findUnique({ where: { id: node.id } });
    expect(after?.powerState).toBe("waking");
  });

  it("returns 409 when no MAC has been captured", async () => {
    const node = await prisma.node.create({
      data: { name: "spark-test", ipAddress: "192.168.44.41", powerState: "off" },
    });
    const res = await request(makeApp(vi.fn())).post(`/api/nodes/${node.id}/wake`).send();
    expect(res.status).toBe(409);
  });

  it("returns 404 for an unknown node", async () => {
    const res = await request(makeApp(vi.fn())).post(`/api/nodes/nope/wake`).send();
    expect(res.status).toBe(404);
  });
});
```

- [ ] **Step 2: Run it to verify it fails**

Run: `npx vitest run packages/server/src/__tests__/integration/nodes.wake.test.ts`
Expected: FAIL — POST returns 404 (route absent).

- [ ] **Step 3: Implement the route**

In `packages/server/src/routes/nodes.ts`, add the import near the others:

```ts
import { broadcastFor, sendMagicPacket as defaultWolSend } from "../nodes/wol.js";
```

Add the route after `POST /:id/power`:

```ts
// POST /api/nodes/:id/wake — send a Wake-on-LAN magic packet to a powered-off node.
nodesRouter.post("/:id/wake", async (req, res) => {
  const node = await prisma.node.findUnique({ where: { id: req.params.id } });
  if (!node) return res.status(404).json({ error: "Node not found" });
  if (!node.macAddress) {
    return res.status(409).json({
      error: "No MAC captured for this node yet — it must have been audited or shut down via the manager at least once.",
    });
  }
  if (!node.ipAddress) return res.status(400).json({ error: "Node has no ipAddress" });

  const wolSend = (req.app.get("wolSend") || defaultWolSend) as typeof defaultWolSend;
  try {
    await wolSend(node.macAddress, broadcastFor(node.ipAddress));
  } catch (err) {
    return res.status(502).json({ error: `WOL send failed: ${String(err)}` });
  }

  await prisma.node.update({ where: { id: node.id }, data: { powerState: "waking" } });
  sseBroadcast({ type: "node:status", payload: { nodeId: node.id, powerState: "waking" } });
  res.json({ status: "ok", powerState: "waking" });
});
```

- [ ] **Step 4: Wire the default in index.ts**

In `packages/server/src/index.ts`, near `app.set("sshExec", ...)`, add:

```ts
import { sendMagicPacket } from "./nodes/wol.js";
// ...
app.set("wolSend", sendMagicPacket);
```

- [ ] **Step 5: Run the test to verify it passes**

Run: `npx vitest run packages/server/src/__tests__/integration/nodes.wake.test.ts`
Expected: PASS (3/3).

- [ ] **Step 6: Commit**

```bash
git add packages/server/src/routes/nodes.ts packages/server/src/index.ts packages/server/src/__tests__/integration/nodes.wake.test.ts
git commit -m "feat(server): POST /api/nodes/:id/wake (Wake-on-LAN)"
```

---

### Task 10: Phase 2 full test run + the WOL hardware test round (DECISION GATE)

- [ ] **Step 1: Run the whole suite**

Run: `npm test`
Expected: all green.

- [ ] **Step 2: WOL test round on real hardware**

Rebuild + run the app (CLAUDE.md command). Then, on a recoverable test node:
1. Ensure the node has a MAC stored (audit it, or shut it down once via the manager — both capture the MAC).
2. Click **Shutdown**, confirm → card goes inactive ("Powered off").
3. Wait until the node is fully off (no ping).
4. Click **Wake** → observe whether the node powers on within ~30–60 s (agent reconnects, card returns to online).
5. Repeat 2–3 times for reliability.

- [ ] **Step 3: Record the result and decide**

Write the outcome in this plan file (or a short `docs/...-wol-test-results.md`):
- **If WOL reliably wakes the node from full power-off → STOP. Phase 3 is not needed.** Skip to "Finishing".
- **If WOL does NOT wake the node** (no power-on, or only intermittently), capture the symptom (does the magic packet leave the manager container at all? try sending from a peer node's shell with `wakeonlan <mac>` to isolate Docker-broadcast vs NIC/BIOS support) and **proceed to Phase 3**.

> Likely failure modes to note while diagnosing: (a) the magic packet never leaves the Docker bridge network → would also fail from the container but succeed from the host/peer; (b) the NIC/BIOS does not support WOL from S5 → fails even from a peer on the same switch. Case (b) is what Phase 3 works around by waking from suspend (S3) instead.

---

## Phase 3 — CONTINGENCY: Suspend + Wake-from-Sleep (execute ONLY if Task 10 decided WOL-from-off fails)

The pure `powerCommand("sleep")` (→ `sudo systemctl suspend`) and the `sleep` branch of `POST /:id/power` (→ `powerState: "asleep"`, MAC captured) already exist from Phase 1. This phase exposes a **Sleep** button and re-runs the wake test against a *suspended* node (S3), where WOL is far more commonly supported than from full power-off (S5).

### Task 11: Add a Sleep button to the node card

**Files:**
- Modify: `packages/dashboard/components/node-card.tsx`

- [ ] **Step 1: Add the Sleep action + button**

In `node-card.tsx`, extend the `power` helper's type to include `"sleep"` and add a Sleep button next to Reboot/Shutdown (in the same online-only `<>...</>` group from Task 6):

```ts
async function power(action: "reboot" | "shutdown" | "sleep") {
  const verb = action === "reboot" ? "Reboot" : action === "sleep" ? "Suspend" : "Shut down";
  if (!window.confirm(`${verb} node "${node.name}"? This runs sudo on the machine and will drop its agent.`)) {
    return;
  }
  try {
    await apiFetch(`/api/nodes/${node.id}/power`, { method: "POST", body: JSON.stringify({ action }) });
  } catch (e) {
    window.alert(`Power action failed: ${e instanceof Error ? e.message : String(e)}`);
  }
}
```

Add inside the online `<>...</>` button group:

```tsx
<button
  onClick={() => power("sleep")}
  className="text-[10px] px-2 py-0.5 rounded bg-indigo-900/60 text-indigo-300 hover:bg-indigo-800"
>
  Sleep
</button>
```

The existing `isOff` check already treats `"asleep"` as inactive and shows the **Wake** button, so no other UI change is needed.

- [ ] **Step 2: Build the dashboard**

Run: `npm run build -w packages/dashboard`
Expected: succeeds.

- [ ] **Step 3: Commit**

```bash
git add packages/dashboard/components/node-card.tsx
git commit -m "feat(dashboard): Sleep (suspend) button for wake-from-S3 contingency"
```

---

### Task 12: Wake-from-suspend test round (DECISION GATE)

- [ ] **Step 1: Suspend + wake test on real hardware**

On a recoverable test node:
1. Click **Sleep**, confirm → node enters S3, card shows "Powered off", `powerState = asleep`, MAC captured.
2. Confirm the node is suspended (mgmt IP stops responding to ping / SSH).
3. Click **Wake** → observe whether the node resumes within ~10–30 s (agent reconnects, card returns to online).
4. Repeat 2–3 times.

- [ ] **Step 2: Record the result**

- **If wake-from-suspend works** → document it as the supported recovery path for this cluster (S3 + WOL), and note in the README/SELF-HOSTING docs that full power-off requires manual/IPMI recovery while Sleep+Wake is remote-recoverable.
- **If wake-from-suspend also fails** → document that this hardware does not support remote wake on either path; recommend leaving the Sleep/Wake buttons hidden behind a config flag, or removing the Wake button and keeping only Reboot/Shutdown (which always work). File a follow-up for IPMI/BMC-based power control if the boxes have a BMC.

- [ ] **Step 3: Update docs to reflect the supported path**

Edit `docs/SELF-HOSTING.md` (and `README.md` if it lists features) with a short "Node power management" subsection stating exactly which actions are supported and how recovery works on this hardware, per the test results.

```bash
git add docs/SELF-HOSTING.md README.md
git commit -m "docs: record supported node power/wake recovery path"
```

---

## Finishing

- [ ] Run `npm test` one final time — all green.
- [ ] Use **superpowers:finishing-a-development-branch** to merge / open a PR.
- [ ] In the PR description, state explicitly which phases were executed, the WOL/suspend test results, and any actions (e.g. WOL works / only S3 wake works / no remote wake — Reboot+Shutdown only).

---

## Self-Review (completed)

**Spec coverage:**
- Reboot button + confirmation per node → Task 6 (button) + Task 3 (endpoint, `--no-block reboot`). ✓
- Shutdown button + confirmation per node → Task 6 + Task 3 (`--no-block poweroff`). ✓
- Confirmation names the specific node → Task 6 `window.confirm(`... node "${node.name}" ...`)`. ✓
- Shut-down nodes stay in the list but inactive → list is DB-driven (already), `powerState=off` + dim/Powered-off rendering (Task 6), and "off" survives agent disconnect (Task 4 deliberately does NOT touch the close handler). ✓
- WOL to wake sleeping nodes → Tasks 8–9 (helpers + endpoint), Task 6 Wake button. ✓
- "If WOL not working after a round of tests, replace with sleep + wake-from-sleep test" → Task 10 decision gate → Phase 3 (Tasks 11–12), with the `sleep`/suspend path already wired in Phase 1. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"/"similar to" — every code step shows complete code and exact commands. ✓

**Type consistency:** `powerState` values (`on`/`rebooting`/`off`/`waking`/`asleep`) are used consistently across schema (Task 1), endpoint (Task 3/9), agent-hub reset (Task 4), and card `isOff` check (Task 6). `PowerAction` (`reboot`/`shutdown`/`sleep`) matches between `power.ts`, the endpoint validation, and the card's `power()` helper. `sshExec` / `wolSend` injection keys match between `index.ts` defaults and the route `req.app.get(...)` reads and the test `app.set(...)` stubs. ✓

**Open assumptions flagged in-plan:** passwordless sudo for poweroff/reboot; /24 default subnet for `broadcastFor` (override prefix if your cluster differs); Docker-container WOL broadcast reachability and S5/S3 WOL support are deliberately deferred to the Task 10/12 hardware test rounds rather than assumed.
