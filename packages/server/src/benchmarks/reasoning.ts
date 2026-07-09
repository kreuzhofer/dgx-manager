// Strip a reasoning model's thinking from a chat-completion answer so lm-eval's
// answer-extraction filters see only the final answer. Handles: complete
// <think>…</think> blocks; a template-injected open tag where only </think> is
// echoed (keep text after the last close); and a truncated open tag with no
// close (no answer was produced → empty). No-op when there are no tags, so it is
// safe for non-reasoning models.
export function stripReasoning(content: string): string {
  if (!content) return content;
  let out = content.replace(/<think>[\s\S]*?<\/think>/gi, "");

  const lower = out.toLowerCase();
  const lastClose = lower.lastIndexOf("</think>");
  if (lastClose !== -1) {
    out = out.slice(lastClose + "</think>".length);
  }

  const openIdx = out.toLowerCase().indexOf("<think>");
  if (openIdx !== -1) {
    out = out.slice(0, openIdx);
  }
  return out.trim();
}
