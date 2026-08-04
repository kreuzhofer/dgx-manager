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
 * Shell that writes the drop-in body to stdout, for the caller to redirect.
 *
 * Reads `$OLLAMA_PREEXISTING` (`"0"` = the manager installed this Ollama) from
 * the surrounding script.
 */
export function ollamaDropInBody(opts: OllamaDropInOptions): string {
  const { userExpr, storageExpr } = opts;
  const sudo = opts.sudo ?? "";
  return `{
  echo "[Service]"
  if [ "$OLLAMA_PREEXISTING" = "0" ]; then
    echo "User=${userExpr}"
    echo "Environment=HOME=/home/${userExpr}"
  fi
  echo "Environment=OLLAMA_HOST=0.0.0.0"
  echo "Environment=OLLAMA_MAX_LOADED_MODELS=0"
  echo "Environment=OLLAMA_KEEP_ALIVE=-1"
  if [ "$OLLAMA_PREEXISTING" = "0" ] && mountpoint -q "${storageExpr}"; then
    echo "Environment=OLLAMA_MODELS=${storageExpr}/models/ollama"
    ${sudo}mkdir -p "${storageExpr}/models/ollama"
    ${sudo}chown -R "${userExpr}":"${userExpr}" "${storageExpr}/models/ollama" 2>/dev/null || true
  fi
}`;
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
