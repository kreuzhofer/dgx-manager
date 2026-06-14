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
