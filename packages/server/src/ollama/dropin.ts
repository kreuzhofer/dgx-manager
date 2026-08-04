/**
 * The Ollama systemd drop-in policy, rendered as shell.
 *
 * Two paths configure Ollama on a node — the SSH provisioner and the token
 * install script — and they must impose the same policy. Expressed separately
 * they have drifted twice: once leaving a pre-existing Ollama on its default
 * loopback bind (agenthost, 2026-07-28), and once claiming `User`/`HOME`/
 * `OLLAMA_MODELS` from an Ollama the manager did not install, which orphans
 * whatever model store that service already has (#11). Both paths now render
 * this body, so a change to the policy can only be made in one place.
 *
 * The policy itself:
 *
 *   - Settings that govern how the service is *reached* and how it manages
 *     memory are safe to impose on any Ollama, and are applied to all of them.
 *   - `User`, `HOME` and `OLLAMA_MODELS` decide where the service's data lives,
 *     so they are claimed ONLY for an Ollama the manager installed itself.
 *     Adopting someone else's service means adopting where it keeps its models.
 *   - `OLLAMA_MODELS` additionally requires shared storage to be mounted:
 *     pointing it at an absent path hides every model just as effectively.
 *
 * The caller owns everything around the body — detecting a pre-existing
 * install, comparing before restarting, and the service's boot state.
 */

export interface OllamaDropInOptions {
  /**
   * Shell expression for the user the service should run as, e.g. `daniel` or
   * `${AGENT_USER}`. Only used in the manager-installed branch.
   */
  userExpr: string;
  /**
   * Shell expression for the shared-storage root, e.g. `/mnt/tank` or
   * `$STORAGE`. Pass it bare — this renderer adds the quoting, so a path
   * containing a space cannot word-split.
   */
  storageExpr: string;
  /** Prefix for privileged commands: `"sudo "` over SSH, `""` when already root. */
  sudo?: string;
}

/**
 * Marker recording that the manager installed this Ollama, and may therefore
 * decide where its data lives.
 *
 * Ownership has to be a fact we *record*, not a state we infer. The obvious
 * inference — "the unit exists, so it was already here" — is wrong the moment
 * the manager installs Ollama itself: on the next run its own service looks
 * foreign, and the ownership settings are stripped back off. Worse, an earlier
 * form of this check also required the unit to be *enabled*, which fleet policy
 * turns off, so a node the manager had just provisioned classified as someone
 * else's on the very next run.
 */
export const OLLAMA_MANAGED_MARKER = ".installed-by-dgx-manager";

/**
 * Shell that writes the drop-in body to stdout, for the caller to redirect.
 *
 * Reads `$OLLAMA_MANAGED` (`"1"` = the manager installed this Ollama, so it may
 * claim the ownership settings) from the surrounding script.
 */
export function ollamaDropInBody(opts: OllamaDropInOptions): string {
  const { userExpr, storageExpr } = opts;
  const sudo = opts.sudo ?? "";
  return `{
  echo "[Service]"
  if [ "$OLLAMA_MANAGED" = "1" ]; then
    echo "User=${userExpr}"
    echo "Environment=HOME=/home/${userExpr}"
  fi
  echo "Environment=OLLAMA_HOST=0.0.0.0"
  echo "Environment=OLLAMA_MAX_LOADED_MODELS=0"
  echo "Environment=OLLAMA_KEEP_ALIVE=-1"
  if [ "$OLLAMA_MANAGED" = "1" ] && mountpoint -q "${storageExpr}"; then
    echo "Environment=OLLAMA_MODELS=${storageExpr}/models/ollama"
    ${sudo}mkdir -p "${storageExpr}/models/ollama"
    ${sudo}chown -R "${userExpr}":"${userExpr}" "${storageExpr}/models/ollama" 2>/dev/null || true
  fi
}`;
}

/**
 * Shell that decides whether the manager owns this node's Ollama, installing it
 * when absent. Sets `$OLLAMA_MANAGED` for the body above.
 *
 * Presence is tested with `list-unit-files`, which succeeds whether the unit is
 * enabled or disabled — fleet policy disables Ollama, so `is-enabled` would
 * report every node the manager provisioned as foreign.
 */
export function ollamaOwnershipProbe(opts: { markerDirExpr: string; sudo?: string; installCmd: string }): string {
  const sudo = opts.sudo ?? "";
  const marker = `${opts.markerDirExpr}/${OLLAMA_MANAGED_MARKER}`;
  return `OLLAMA_MARKER="${marker}"
if systemctl list-unit-files ollama.service >/dev/null 2>&1 || command -v ollama >/dev/null 2>&1; then
  : # already present — ours only if we recorded it as such
else
  ${opts.installCmd}
  ${sudo}mkdir -p "${opts.markerDirExpr}"
  ${sudo}touch "$OLLAMA_MARKER"
fi
if [ -e "$OLLAMA_MARKER" ]; then OLLAMA_MANAGED=1; else OLLAMA_MANAGED=0; fi`;
}

/**
 * Environment settings imposed on every Ollama, whoever installed it.
 *
 * Exported so both paths' tests can assert against one list rather than each
 * repeating the strings — a drifting test is how the policy drifted before.
 */
export const OLLAMA_UNIVERSAL_SETTINGS = [
  // Bind on all interfaces. A pre-existing Ollama keeps its default loopback
  // bind, which silently makes the node unreachable by the manager while the
  // deployment still reports "running" (agenthost, 2026-07-28).
  "Environment=OLLAMA_HOST=0.0.0.0",
  "Environment=OLLAMA_MAX_LOADED_MODELS=0",
  // Never unload an idle model. The manager is the only host Ollama's firewall
  // admits, and the gateway does not route to a deployment that is not serving
  // — so an evicted model has nothing left that could reload it and stays dark
  // until someone restarts the deployment by hand. Resident memory is the
  // deliberate trade.
  "Environment=OLLAMA_KEEP_ALIVE=-1",
] as const;

/** Settings that decide where the service's data lives — never imposed on an
 *  Ollama the manager did not install. */
export const OLLAMA_OWNERSHIP_SETTINGS = ["User=", "Environment=HOME=", "Environment=OLLAMA_MODELS="] as const;
