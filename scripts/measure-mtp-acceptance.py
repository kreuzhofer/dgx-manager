#!/usr/bin/env python3
"""Measure MTP speculative-decoding acceptance on a serving deployment.

Acceptance is the figure that decides whether a drafter is worth realigning, and
it is easy to overstate by a third. This script exists because getting it right
took three attempts; the trap is documented under "RUNAWAYS" below.

    ./scripts/measure-mtp-acceptance.py --node http://192.168.44.36:8000 \
                                        --model chat3d-judge-rc0

Two numbers are reported, and they are not interchangeable:

    acceptance rate   = accepted / drafted            fraction of proposals kept
    acceptance length = 1 + accepted / drafts         tokens emitted per
                        verification step; the 1 is the target model's own
                        always-correct token. This is the throughput multiplier.

MEASURED REFERENCES on this fleet, for judging a new number (2026-09-25):

                        length   rate    per-position %
    qwen3.8-27b-nvfp4    4.39   68.3%   86.9/75.4/65.9/58.3/52.2   aligned drafter
    chat3d-judge-rc0     2.93   42.9%   70.4/57.4/31.8/20.7/13.1   base head on a
                                                                   merged trunk

The gap is the cost of a transplanted head: rc0's drafter proposes from a
representation it was never fitted to (cosine 0.79 between base and merged final
hidden states). It costs throughput only — the target model verifies every draft,
so a stale drafter can never change the output under greedy decoding.

RUNAWAYS — the trap, and why this script scrapes per call.

A generation that runs to `max_tokens` is usually a repetitive loop, and
repetitive text drafts far too easily. On the same deployment, the same probe gave:

    one runaway supplying 87% of generated tokens   -> 3.74
    capped at 600 tokens, 39% contamination         -> 3.16
    runaways excluded per call                      -> 2.93

The excluded call's own acceptance was 3.47 against the clean 2.93 — the effect
measuring itself. Aggregate before/after counter deltas CANNOT separate the
degenerate call from the rest, so this script scrapes `/metrics` around every call
and drops any whose `finish_reason` is `"length"`. The tell in contaminated data is
the per-position profile: a cliff rather than a smooth decay.

If the deployment has no `vllm:spec_decode_*` metrics, MTP is not enabled — check
the recipe carries `--speculative-config`.
"""
import argparse
import base64
import json
import re
import statistics
import struct
import sys
import time
import urllib.request
import zlib

VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {"type": "string", "enum": ["pass", "fail", "borderline"]},
        "score": {"type": "integer", "minimum": 0, "maximum": 10},
        "failures": {"type": "array", "items": {"type": "string"}, "maxItems": 5},
        "rationale": {"type": "string"},
    },
    "required": ["verdict", "score", "failures", "rationale"],
    "additionalProperties": False,
}

JUDGE_PROMPT = (
    "You are judging whether a generated 3D model matches its prompt. "
    "Renders of the candidate are attached, from canonical viewpoints. "
    "Assess silhouette fidelity, proportion, surface continuity, and whether any "
    "requested feature is missing or duplicated. Return the verdict object."
)

TEXT_PROMPT = (
    "Explain, in a single paragraph of ordinary prose, why memory bandwidth rather "
    "than arithmetic throughput sets the speed of autoregressive decoding."
)


def png(edge: int, seed: int) -> bytes:
    """A deterministic non-uniform PNG, hand-written to avoid a Pillow dependency.

    Non-uniform deliberately: a flat colour compresses to nothing and understates
    the bytes on the wire (though not the token count, which follows the patch
    grid).
    """
    rows = bytearray()
    for y in range(edge):
        rows.append(0)  # filter type 0
        for x in range(edge):
            rows += bytes(((x * 7 + seed * 31) % 256,
                           (y * 5 + seed * 17) % 256,
                           ((x ^ y) + seed * 11) % 256))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (struct.pack(">I", len(data)) + tag + data
                + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF))

    return (b"\x89PNG\r\n\x1a\n"
            + chunk(b"IHDR", struct.pack(">IIBBBBB", edge, edge, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(bytes(rows), 6))
            + chunk(b"IEND", b""))


def scrape(node: str):
    """The spec-decode counters, plus per-draft-position acceptance."""
    raw = urllib.request.urlopen(f"{node}/metrics", timeout=30).read().decode()
    totals, per_pos = {}, {}
    for line in raw.splitlines():
        if line.startswith("#"):
            continue
        for key in ("num_drafts_total", "num_draft_tokens_total",
                    "num_accepted_tokens_total"):
            if line.startswith(f"vllm:spec_decode_{key}"):
                totals[key] = float(line.rsplit(" ", 1)[1])
        m = re.match(r'vllm:spec_decode_num_accepted_tokens_per_pos_total'
                     r'\{.*position="(\d+)".*\}\s+(\S+)', line)
        if m:
            per_pos[int(m.group(1))] = float(m.group(2))
    return totals, per_pos


def one_call(args, images, note):
    if images:
        content = [{"type": "text", "text": JUDGE_PROMPT + note}]
        content += [{"type": "image_url",
                     "image_url": {"url": f"data:image/png;base64,{b}"}} for b in images]
        body = {"messages": [{"role": "user", "content": content}],
                "response_format": {"type": "json_schema",
                                    "json_schema": {"name": "verdict",
                                                    "schema": VERDICT_SCHEMA,
                                                    "strict": True}}}
    else:
        body = {"messages": [{"role": "user", "content": TEXT_PROMPT + note}]}
    body |= {"model": args.model, "max_tokens": args.max_tokens, "temperature": 0}
    if args.thinking_kwargs:
        body["chat_template_kwargs"] = json.loads(args.thinking_kwargs)

    req = urllib.request.Request(f"{args.node}/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    with urllib.request.urlopen(req, timeout=args.timeout) as r:
        payload = json.loads(r.read())
    return (time.perf_counter() - t0,
            payload["usage"]["completion_tokens"],
            payload["usage"]["prompt_tokens"],
            payload["choices"][0].get("finish_reason"))


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--node", required=True, help="http://<ip>:<port> of the deployment")
    p.add_argument("--model", required=True, help="served model name")
    p.add_argument("--reps", type=int, default=10)
    p.add_argument("--images", type=int, default=8, help="0 for a text-only workload")
    p.add_argument("--px", type=int, default=768, help="square render edge")
    p.add_argument("--max-tokens", type=int, default=600,
                   help="keep this near real output length; a high cap invites "
                        "runaways, which this script excludes but cannot un-run")
    p.add_argument("--thinking-kwargs", default='{"enable_thinking": false, "thinking": false}',
                   help="chat_template_kwargs as JSON; '' to send none")
    p.add_argument("--timeout", type=int, default=1800,
                   help="a 300s default bounds the WHOLE generation and has "
                        "livelocked harnesses on this fleet")
    args = p.parse_args()

    sets = {r: [base64.b64encode(png(args.px, r * 100 + i)).decode()
                for i in range(args.images)] for r in range(args.reps + 1)}

    # A cold model pays JIT and cache-warm costs no caller sees twice.
    print("warm-up (untimed, excluded from the counters) ...", flush=True)
    one_call(args, sets[0], " Item 0.")

    if not scrape(args.node)[0]:
        sys.exit("no vllm:spec_decode_* metrics — MTP is not enabled on this "
                 "deployment (does the recipe carry --speculative-config?)")

    rows = []
    for r in range(1, args.reps + 1):
        before, pos_before = scrape(args.node)
        wall, ctok, ptok, fin = one_call(args, sets[r], f" Item {r}.")
        after, pos_after = scrape(args.node)
        drafts = after["num_drafts_total"] - before["num_drafts_total"]
        dtok = after["num_draft_tokens_total"] - before["num_draft_tokens_total"]
        acc = after["num_accepted_tokens_total"] - before["num_accepted_tokens_total"]
        if drafts <= 0:
            print(f"  rep {r:2}: no drafts recorded — skipping"); continue
        rows.append({"rep": r, "tok": ctok, "wall": wall, "fin": fin, "drafts": drafts,
                     "dtok": dtok, "acc": acc, "length": 1 + acc / drafts,
                     "rate": acc / dtok if dtok else 0.0,
                     "pos": {k: pos_after[k] - pos_before.get(k, 0.0) for k in pos_after}})
        print(f"  rep {r:2}: {ctok:4} tok {wall:6.1f}s {ctok/wall:5.1f} tok/s | "
              f"len {rows[-1]['length']:.2f} rate {100*rows[-1]['rate']:4.1f}% | {fin}"
              + ("  <- EXCLUDED (runaway)" if fin == "length" else ""), flush=True)

    clean = [r for r in rows if r["fin"] != "length"]
    runaway = [r for r in rows if r["fin"] == "length"]
    if not clean:
        sys.exit("every rep hit the token cap — lower --max-tokens, or the model "
                 "is looping on this workload")

    D = sum(r["drafts"] for r in clean)
    T = sum(r["dtok"] for r in clean)
    A = sum(r["acc"] for r in clean)
    lens = [r["length"] for r in clean]
    tput = [r["tok"] / r["wall"] for r in clean]

    print(f"\n=== {len(clean)} clean reps, {len(runaway)} runaway excluded ===")
    print(f"drafts {D:,.0f}   draft tokens {T:,.0f} ({T/D:.2f} per draft)   accepted {A:,.0f}")
    print(f"ACCEPTANCE LENGTH  {1 + A/D:.2f}   (aligned reference 4.39, transplanted 2.93)")
    print(f"ACCEPTANCE RATE    {100*A/T:.1f}%  (aligned reference 68.3%, transplanted 42.9%)")
    if len(lens) > 1:
        print(f"per-rep length: min {min(lens):.2f} max {max(lens):.2f} "
              f"stdev {statistics.stdev(lens):.2f}")
    print("\nper draft position:")
    for pos in sorted({k for r in clean for k in r["pos"]}):
        share = sum(r["pos"].get(pos, 0.0) for r in clean) / D
        print(f"  pos {pos}: {100*share:5.1f}%")
    print(f"\nthroughput: mean {statistics.mean(tput):.1f} tok/s "
          f"({min(tput):.1f}-{max(tput):.1f})")
    if runaway:
        print(f"\nrunaway reps {[r['rep'] for r in runaway]} measured "
              f"{[round(r['length'], 2) for r in runaway]} — higher than the clean "
              f"figure, which is exactly why they are excluded.")


if __name__ == "__main__":
    main()
