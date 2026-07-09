// Strip a reasoning model's thinking from a chat-completion answer so lm-eval's
// answer-extraction filters see only the final answer. Handles: complete
// <think>…</think> blocks; a template-injected open tag where only </think> is
// echoed (keep text after the last close); and a truncated open tag with no
// close (no answer was produced → empty). No-op when there are no tags, so it is
// safe for non-reasoning models.
export function stripReasoning(content: string): string {
  if (!content) return content;
  let out = content.replace(/<think>[\s\S]*?<\/think>/gi, "");

  // Template-injected open tag: only </think> is echoed → keep text after the last close.
  const closes = [...out.matchAll(/<\/think>/gi)];
  if (closes.length > 0) {
    const last = closes[closes.length - 1];
    out = out.slice(last.index! + last[0].length);
  }

  // Truncated mid-think: an open tag with no close → no answer was produced.
  const open = out.match(/<think>/i);
  if (open && open.index !== undefined) {
    out = out.slice(0, open.index);
  }
  return out.trim();
}
