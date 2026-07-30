/**
 * Copy text to the clipboard, including on non-secure origins.
 *
 * The dashboard is served over plain http on the LAN (`http://<ip>:3000`),
 * which is not a secure context, so `navigator.clipboard` is undefined there —
 * the async API silently does nothing and a naive caller reports success it did
 * not have. Falls back to the selection-based copy, and throws when even that
 * fails so the caller can say so rather than claiming it worked.
 *
 * The same workaround predates this in onboarding-command.tsx and
 * claude-launch-modal.tsx; this is its shared home.
 */
export async function copyText(text: string): Promise<void> {
  if (typeof navigator !== "undefined" && navigator.clipboard && window.isSecureContext) {
    await navigator.clipboard.writeText(text);
    return;
  }

  const ta = document.createElement("textarea");
  ta.value = text;
  ta.setAttribute("readonly", "");
  ta.style.position = "fixed";
  ta.style.top = "0";
  ta.style.left = "0";
  ta.style.opacity = "0";
  document.body.appendChild(ta);
  ta.focus();
  ta.select();
  const ok = document.execCommand("copy");
  document.body.removeChild(ta);
  if (!ok) throw new Error("execCommand('copy') returned false");
}
