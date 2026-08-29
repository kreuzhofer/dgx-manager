#!/usr/bin/env python3
"""Needle-in-a-haystack probe: does a deployment actually serve its window?

Written for issue #24 and referenced by
recipes/dgxrun/glm-5.3-flash-nvfp4-2x.yaml, which forbids raising
`max_model_len` without evidence from this script.

The reason it exists: a vLLM deployment can boot, size its KV cache, pass
/health and answer short prompts correctly while being completely unable to
serve the context window it advertises. GLM-5.3-Flash at max_model_len 491520
did exactly that, dying on its first long prefill inside a GB10 kernel limit.
Nothing short of a real long prompt tells the two apart.

Builds a prompt at a target token count, buries a fact at a given depth, and
checks it comes back. Reports the server's OWN prompt_tokens, because a
chars-per-token estimate is not trustworthy — this filler runs ~5.31
chars/token, and assuming ~4 made one probe look 33% larger than it was.

CAUTION when reading the throughput it prints: re-running the same prompt hits
the prefix cache and reports an order of magnitude more tok/s than a real cold
prefill. Only the first run against a fresh deployment is a prefill measurement.

  python3 scripts/needle-probe.py <target_tokens> [depth_fraction] [endpoint]

Exits 0 only if the needle came back AND finish_reason was "stop".
"""
import json
import sys
import time
import urllib.request

target = int(sys.argv[1]) if len(sys.argv) > 1 else 236000
depth = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
url = sys.argv[3] if len(sys.argv) > 3 else "http://192.168.44.37:8000/v1/chat/completions"

NEEDLE = "The maintenance passphrase for the Kestrel relay is FERROUS-ORCHID-2291."
QUESTION = "What is the maintenance passphrase for the Kestrel relay? Reply with only the passphrase."

# ~4 chars/token for this filler; deliberately varied so it does not compress
# into a trivially cacheable repeat.
FILLER = (
    "Routine telemetry from the outer survey array indicates nominal drift across "
    "all monitored channels, with periodic recalibration logged by the duty officer. "
)
per = len(FILLER) // 4
n = max(1, target // per)
body = []
for i in range(n):
    body.append(f"[entry {i:06d}] {FILLER}")
at = int(len(body) * depth)
body.insert(at, f"[entry {at:06d}] {NEEDLE} ")
haystack = "".join(body)

payload = {
    "model": "glm-5.3-flash",
    "messages": [{"role": "user", "content": haystack + "\n\n" + QUESTION}],
    "max_tokens": 120,
    "temperature": 0.0,
}
data = json.dumps(payload).encode()
print(f"target≈{target} tok | chars={len(haystack):,} | depth={depth:.0%} | body={len(data)/1e6:.1f} MB")

t0 = time.time()
try:
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        out = json.load(r)
except Exception as e:
    print(f"REQUEST FAILED after {time.time()-t0:.0f}s: {type(e).__name__}: {e}")
    sys.exit(2)
el = time.time() - t0

if "error" in out and out["error"]:
    print(f"API ERROR after {el:.0f}s: {json.dumps(out['error'])[:400]}")
    sys.exit(2)

ch = out["choices"][0]
text = (ch["message"].get("content") or "").strip()
usage = out.get("usage", {})
pt = usage.get("prompt_tokens")
found = "FERROUS-ORCHID-2291" in text.upper()
print(f"prompt_tokens={pt} completion={usage.get('completion_tokens')} "
      f"finish={ch.get('finish_reason')} elapsed={el:.0f}s "
      f"prefill≈{(pt/el if pt else 0):.0f} tok/s")
print(f"NEEDLE {'FOUND' if found else 'NOT FOUND'} | reply: {text[:160]!r}")
sys.exit(0 if (found and ch.get("finish_reason") == "stop") else 1)
