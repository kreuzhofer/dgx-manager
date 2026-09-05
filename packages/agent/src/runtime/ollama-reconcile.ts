/**
 * What to do with a persisted Ollama deployment when the agent reconnects.
 *
 * Ollama deployments were the only kind the agent did not reconcile. Sparkrun
 * and dgxrun both reattach from the deployment store; Ollama tracked its
 * deployments in memory only, so an agent restart — or a node reboot — left the
 * manager believing a deployment was serving while nothing on the node was.
 *
 * That was survivable only because Ollama used to start at boot: the service
 * came back on its own and loaded the model on demand for the first request.
 * Fleet policy is autostart *off* (the `:11434` API is unauthenticated and the
 * agent's firewall is applied at agent start, so an auto-started Ollama is
 * briefly exposed), and once both paths honour that, nothing restores the
 * deployment. Hence this.
 *
 * Pure so the decision can be stated as an invariant; the caller does the IO.
 */

export interface OllamaReconcileState {
  /** Is the Ollama service answering on its port? */
  serviceRunning: boolean;
  /** Is this deployment's model resident in memory? */
  modelLoaded: boolean;
  /** Has an undeploy already begun for this deployment? */
  stopping?: boolean;
}

export type OllamaReconcileAction =
  /** Start the service so the deployment can serve again. */
  | { kind: "restore"; reason: string }
  /** Service is up and the model is resident. */
  | { kind: "serving"; reason: string }
  /** Service is up, model unloaded — normal, Ollama loads on demand. */
  | { kind: "idle"; reason: string }
  /** Leave it alone entirely. */
  | { kind: "skip"; reason: string };

export function reconcileOllamaAction(state: OllamaReconcileState): OllamaReconcileAction {
  // An undeploy already in flight wins over everything: a reconnect landing
  // mid-teardown must not bring the deployment back.
  if (state.stopping) return { kind: "skip", reason: "undeploy in progress" };

  if (!state.serviceRunning) return { kind: "restore", reason: "Ollama is not running" };

  if (!state.modelLoaded) {
    return { kind: "idle", reason: "model not resident; Ollama loads it on demand" };
  }

  return { kind: "serving", reason: "model resident" };
}

/** Inputs to the retry decision below. */
export interface OllamaRetryState {
  /** Deployments whose restore failed and that no one has re-checked since. */
  pendingRestores: number;
  /** Is a reconcile running right now? */
  inFlight: boolean;
}

/**
 * Should the health tick run the reconcile again?
 *
 * The reconcile above only runs when the agent registers over the WebSocket.
 * That made a restore failure permanent: on the agenthost reboot of
 * 2026-09-04 the restore lost a race against `network-online.target`, the
 * deployment was reported `failed`, the agent then stayed connected — so no
 * second reconcile ever ran, even though Ollama came up 76s later and the
 * health tick was polling it successfully the whole time.
 *
 * Driving a retry from the health tick makes that transient failure heal
 * itself. The in-flight guard is load-bearing: a reconcile can wait minutes on
 * a slow boot while the tick fires every 15s, and overlapping reconciles would
 * each issue their own `systemctl start` and status report.
 */
export function shouldRetryOllamaReconcile(state: OllamaRetryState): boolean {
  return state.pendingRestores > 0 && !state.inFlight;
}
