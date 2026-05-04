"use client";

import { useEffect, useRef, useState } from "react";

/**
 * Tail-following log viewer.
 *
 * Behavior contract (kept consistent across deployments + finetune pages):
 *   - On open / mount, the viewer is auto-scrolled to the bottom AND enters
 *     "follow" mode. New log lines arriving while in follow mode keep the
 *     viewer pinned to the tail.
 *   - The user can scroll up to read history. Scrolling more than ~40 px
 *     away from the bottom disables follow mode (so new lines don't yank
 *     them back down). Scrolling back to within 40 px re-enables it.
 *   - When follow mode is OFF, a "⬇ Follow" button appears bottom-right.
 *     Clicking it jumps to the bottom and re-enables follow mode without
 *     having to manually scroll the whole way.
 *
 * Replaces the older global-querySelectorAll + "if near bottom, snap"
 * effect that was shared across pages — that pattern broke re-opening a
 * viewer (mount left scrollTop=0, distance-from-bottom >> threshold, so
 * auto-scroll never fired and the viewer stayed stuck at line 1).
 */
export function LogViewer({
  content,
  className,
}: {
  content: string;
  className?: string;
}) {
  const ref = useRef<HTMLPreElement>(null);
  const [following, setFollowing] = useState(true);

  // On mount, snap to the bottom + start following. The empty deps are
  // intentional — we only want this on the first render of a particular
  // viewer instance (i.e. when the user opens the panel).
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, []);

  // While following, keep pinned to the tail on every content change.
  useEffect(() => {
    if (!following) return;
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [content, following]);

  const onScroll = (e: React.UIEvent<HTMLPreElement>) => {
    const el = e.currentTarget;
    // 40 px tolerance for browser scroll-rounding so we don't bounce out of
    // follow mode just because the user nudged the wheel by a single tick.
    const distanceFromBottom = el.scrollHeight - el.scrollTop - el.clientHeight;
    setFollowing(distanceFromBottom <= 40);
  };

  const followNow = () => {
    const el = ref.current;
    if (el) el.scrollTop = el.scrollHeight;
    setFollowing(true);
  };

  return (
    <div className="relative">
      <pre
        ref={ref}
        onScroll={onScroll}
        className={
          className ??
          "mt-3 bg-black/50 border border-gray-800 rounded p-3 text-xs text-gray-400 font-mono max-h-64 overflow-y-auto whitespace-pre-wrap"
        }
      >
        {content}
      </pre>
      {!following && (
        <button
          type="button"
          onClick={followNow}
          aria-label="Follow log tail"
          className="absolute bottom-2 right-2 text-xs px-2 py-1 rounded bg-blue-600 hover:bg-blue-500 text-white shadow"
        >
          ⬇ Follow
        </button>
      )}
    </div>
  );
}
