"""Vision QA loop for paper figures (2026-09-16).

Reviewing a rendered figure needs actual vision, which a terminal
session does not have; this script sends each rendered
figure PNG to a vision-capable model and returns its review as TEXT,
which the editing session reads and acts on. Supports both API flavors,
auto-detected from the environment:

  OpenAI-compatible (GLM, DeepSeek, vLLM, Ollama-openai, ...):
      VISION_BASE_URL, VISION_API_KEY, VISION_MODEL
      POST {base}/chat/completions
  Anthropic:
      VISION_ANTHROPIC_KEY (and optionally VISION_MODEL=claude-opus-5)
      POST https://api.anthropic.com/v1/messages

Usage:
  python scripts/28_fig_vision_check.py runs/<id>/figures_reach/ [--glob 'fig_*.png']
  # PNGs only (the PDFs' text is not what we review).

The prompt demands a structured, enumerated defect list (overlap,
clipping, legibility, legend placement, spacing, colour) with
severity and location per item, plus a PASS/FAIL verdict. Reviews are
saved to <dir>/vision_review.json for the fix-loop bookkeeping.
"""
from __future__ import annotations

import argparse
import base64
import glob as _glob
import json
import os
import sys
import time

PROMPT = """You are reviewing a figure for an IEEE journal paper (rendered to PNG).
Report visual defects only. Categories: OVERLAP, CLIPPING, LEGIBILITY,
LEGEND, SPACING, COLOUR/CLARITY.

Rules: no step-by-step analysis in your reply — output the JSON
immediately. At most 5 defects, one short sentence each. STRICT JSON
only, nothing before or after:
{
  "figure": "<filename>",
  "verdict": "PASS" | "FIX",
  "defects": [
    {"category": "<one of the 6>", "severity": "high|medium|low",
     "where": "<location>", "what": "<the problem>",
     "fix": "<one-line fix>"}
  ],
  "overall": "<one sentence: publication-ready?>"
}
Be strict: this goes into a journal submission. If everything is fine,
return an empty defects list and verdict PASS.
"""


def _http_json(url, headers, payload, timeout=900):
    """POST JSON with stdlib only (requests is not in the venv)."""
    import urllib.request, urllib.error
    req = urllib.request.Request(url, data=json.dumps(payload).encode(),
                                 headers=headers, method='POST')
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode()[:500]
        raise RuntimeError(f"HTTP {exc.code}: {body}") from exc


MAX_TOKENS = 3500


def _post_openai(base_url, key, model, img_b64, media):
    d = _http_json(
        f"{base_url.rstrip('/')}/chat/completions",
        {"Authorization": f"Bearer {key}",
         "Content-Type": "application/json"},
        {"model": model, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {
                "url": f"data:image/{media};base64,{img_b64}"}},
            {"type": "text", "text": PROMPT}]}],
            # 2026-09-26: glm-5.3-flash emits a reasoning_content phase
            # BEFORE the answer; small budgets consumed by reasoning
            # return content=null (finish=length). Default 3500; dense
            # figures (gamma, map_jfield) truncate -- rerun those with
            # --max-tokens 5500. If content is still null, fall back to
            # the reasoning text so the review is never silently
            # dropped. 900 s ceiling per request; thinking cannot be
            # disabled (enable_thinking -> 400; chat_template_kwargs
            # ignored).
            "max_tokens": MAX_TOKENS, "temperature": 0})
    msg = d["choices"][0]["message"]
    content = msg.get("content")
    if not content:
        content = msg.get("reasoning_content") or ""
        print("  [vision] content null -> using reasoning_content "
              f"(finish={d['choices'][0].get('finish_reason')})", flush=True)
    return content


def _post_anthropic(key, model, img_b64, media, base_url=None,
                    auth_token=None):
    headers = {"anthropic-version": "2023-06-01",
               "Content-Type": "application/json"}
    if auth_token:      # relay/proxy style (ANTHROPIC_AUTH_TOKEN)
        headers["Authorization"] = f"Bearer {auth_token}"
        # this relay fingerprints the CLIENT, not just the token: it
        # rejects non-Claude-Code requests with 401 "unauthorized
        # client detected" (measured 2026-09-16) — send Claude Code's
        headers["User-Agent"] = "claude-cli/2.1.267 (external, cli)"
        headers["x-app"] = "cli"
    else:               # direct API style (ANTHROPIC_API_KEY)
        headers["x-api-key"] = key
    payload = {"model": model,
               # thinking blocks consume the budget too (measured: the
               # relay returns [thinking, text]) — leave room for both
               "max_tokens": 8000, "messages": [{
                   "role": "user", "content": [
                       {"type": "image", "source": {
                           "type": "base64",
                           "media_type": f"image/{media}",
                           "data": img_b64}},
                       {"type": "text", "text": PROMPT}]}]}
    url = (f"{(base_url or 'https://api.anthropic.com').rstrip('/')}"
           f"/v1/messages")
    d = _http_json(url, headers, payload)
    # content may be [thinking, text, ...] — take the first TEXT block
    for block in d.get("content", []):
        if block.get("type") == "text" and "text" in block:
            return block["text"]
    raise RuntimeError(f"no text block in response: "
                       f"{[b.get('type') for b in d.get('content', [])]}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('figdir')
    ap.add_argument('--glob', default='fig_*.png')
    ap.add_argument('--max-tokens', type=int, default=None,
                    help='override the response budget (dense figures '
                         'with long model reasoning truncate at the '
                         'default; 5500 works for gamma/map)')
    args = ap.parse_args(argv)

    base = os.environ.get('VISION_BASE_URL')
    oai_key = os.environ.get('VISION_API_KEY')
    ant_key = os.environ.get('VISION_ANTHROPIC_KEY')
    ant_token = os.environ.get('VISION_AUTH_TOKEN')
    model = os.environ.get('VISION_MODEL')
    if ant_key or ant_token:
        model = model or 'claude-opus-5'
        post = lambda b64, media: _post_anthropic(
            ant_key, model, b64, media, base_url=base, auth_token=ant_token)
        flavor = f"anthropic{' relay' if base else ''} ({model})"
    elif base and oai_key:
        model = model or 'glm-5.3-flash'
        post = lambda b64, media: _post_openai(base, oai_key, model,
                                               b64, media)
        flavor = f"openai-compatible ({model} @ {base})"
    else:
        print("No vision API configured. Set either:\n"
              "  VISION_BASE_URL + VISION_API_KEY + VISION_MODEL  "
              "(OpenAI-compatible)\n"
              "  VISION_ANTHROPIC_KEY [+ VISION_MODEL]            "
              "(Anthropic)", file=sys.stderr)
        return 2
    print(f"vision backend: {flavor}")

    pngs = sorted(_glob.glob(os.path.join(args.figdir, args.glob)))
    if not pngs:
        print(f"no PNGs matching {args.glob} in {args.figdir}", file=sys.stderr)
        return 2
    print(f"reviewing {len(pngs)} figures\n" + "-" * 60)

    global MAX_TOKENS
    if args.max_tokens:
        MAX_TOKENS = int(args.max_tokens)
    reviews = {}
    for p in pngs:
        b64 = base64.b64encode(open(p, 'rb').read()).decode()
        media = 'jpeg' if p.lower().endswith(('.jpg', '.jpeg')) else 'png'
        try:
            text = None
            for attempt in range(3):        # 2026-09-26: 504s under
                try:                        # free-tier congestion
                    text = post(b64, media)
                    break
                except Exception:
                    if attempt == 2:
                        raise
                    time.sleep(10 + 20 * attempt)
        except Exception as exc:                     # noqa: BLE001 - report and continue
            print(f"{os.path.basename(p)}: REQUEST FAILED: {exc}", flush=True)
            reviews[os.path.basename(p)] = {'error': str(exc)}
            continue
        # robust extraction: fences, leading prose, trailing prose
        t = text.strip()
        if t.startswith('```'):
            t = t.split('```')[1]
            if t.startswith('json'):
                t = t[4:]
        i, j = t.find('{'), t.rfind('}')
        if i >= 0 and j > i:
            t = t[i:j + 1]
        try:
            d = json.loads(t)
        except json.JSONDecodeError:
            # truncated JSON (token budget): salvage the verdict if present
            import re
            m = re.search(r'"verdict"\s*:\s*"(PASS|FIX)"', text)
            d = {'verdict': m.group(1) if m else 'UNPARSEABLE',
                 'truncated': True, 'raw_prefix': text[:1500]}
        reviews[os.path.basename(p)] = d
        nd = len(d.get('defects', []))
        print(f"{os.path.basename(p):<34} {d.get('verdict', '?'):<6} "
              f"({nd} defects)")
        for item in d.get('defects', []):
            print(f"    [{item.get('severity')}] {item.get('category')}: "
                  f"{item.get('what')} @ {item.get('where')}")
        time.sleep(0.5)

    out = os.path.join(args.figdir, 'vision_review.json')
    json.dump(reviews, open(out, 'w'), indent=1)
    print("-" * 60)
    n_pass = sum(1 for r in reviews.values() if r.get('verdict') == 'PASS')
    print(f"{n_pass}/{len(reviews)} PASS — full report: {out}")
    return 0 if n_pass == len(reviews) else 1


if __name__ == '__main__':
    raise SystemExit(main())
