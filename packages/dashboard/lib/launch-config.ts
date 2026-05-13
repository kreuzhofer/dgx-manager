/**
 * Pure helper: parse a FineTuneJob.config JSON blob into an ordered list
 * of {key, label, value} entries for the dashboard "details" panel.
 *
 * The canonical order matches the launch form's field order so the
 * round-trip from "what I clicked" to "what was captured" is obvious.
 * Keys present in config but not in LAUNCH_CONFIG_LABELS are kept at the
 * end with their raw key as the label so we never silently hide data.
 *
 * Defensive against:
 *   - config === null (job has no config)
 *   - config === ""   (older rows, edge case)
 *   - malformed JSON  (returns [] rather than throwing)
 *   - value === null  (JSON-stringified explicit nulls — filtered out)
 *
 * Note: JSON.parse never produces `undefined` property values (undefined
 * is not a JSON value), so there is no explicit undefined-filter.
 */

export const LAUNCH_CONFIG_LABELS: Record<string, string> = {
  learning_rate: "Learning rate",
  batch_size: "Batch size",
  max_seq_length: "Max seq length",
  lora_r: "LoRA rank (r)",
  lora_alpha: "LoRA alpha",
  num_train_epochs: "Epochs",
  max_steps: "Max steps",
};

export interface LaunchConfigEntry {
  key: string;
  label: string;
  value: unknown;
}

export function parseLaunchConfig(raw: string | null | undefined): LaunchConfigEntry[] {
  if (!raw) return [];
  let parsed: Record<string, unknown>;
  try {
    parsed = JSON.parse(raw) as Record<string, unknown>;
  } catch {
    return [];
  }
  if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return [];

  const out: LaunchConfigEntry[] = [];
  for (const key of Object.keys(LAUNCH_CONFIG_LABELS)) {
    if (key in parsed && parsed[key] !== null) {
      out.push({ key, label: LAUNCH_CONFIG_LABELS[key]!, value: parsed[key] });
    }
  }
  for (const key of Object.keys(parsed)) {
    if (key in LAUNCH_CONFIG_LABELS) continue;
    if (parsed[key] === null) continue;
    out.push({ key, label: key, value: parsed[key] });
  }
  return out;
}
