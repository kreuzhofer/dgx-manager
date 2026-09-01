/**
 * Pure decisions about a node's power controls, kept out of the card so they
 * can be tested without a DOM.
 */

/**
 * Whether the Wake button should render for a node.
 *
 * Gated on the captured MAC, not on power state: the MAC is what actually
 * determines whether a magic packet can be addressed at all, and the manager
 * refuses a wake without one. Gating on power state instead would hide Wake on
 * every healthy node — and since an agent reconnect resets powerState to "on",
 * that meant the button was never visible on a running cluster.
 */
export function canWake(macAddress: string | null | undefined): boolean {
  return typeof macAddress === "string" && macAddress.trim() !== "";
}

/**
 * Whether a node is powered down or on its way back up, so the card dims and
 * hides Reboot/Shutdown. "waking" counts: the magic packet went out but the
 * agent has not reconnected yet.
 *
 * Deliberately separate from `canWake` — a node can be active and still be
 * wakeable, since a magic packet aimed at a live host is a no-op.
 */
export function isNodeInactive(powerState: string | null | undefined): boolean {
  return powerState === "off" || powerState === "asleep" || powerState === "waking";
}

/**
 * Copy shown on a card that has no metric samples to draw.
 *
 * Power state chooses the wording; the captured MAC decides whether it can
 * point at the Wake button. Without this second axis an inactive node with no
 * MAC would advertise a button that `canWake` has removed from the card.
 */
export function nodeMetricsPlaceholder(
  powerState: string | null | undefined,
  macAddress: string | null | undefined,
): string {
  if (!isNodeInactive(powerState)) return "No metrics yet";

  const waking = powerState === "waking";
  if (!canWake(macAddress)) {
    const state = waking ? "Waking…" : "Powered off";
    return `${state} — no MAC captured yet, so it can't be woken from here`;
  }
  return waking
    ? "Waking… — click Wake to retry if it doesn't come back"
    : "Powered off — Wake to bring it back";
}
