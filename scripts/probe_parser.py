"""Replicate the claude_assess parser on a representative fenced sample."""

import json

samples = [
    "```json\n[{\"a\": 1}]\n```",
    "```json\n[{\"a\": 1}]\n```\n",
    "```\n[{\"a\": 1}]\n```",
]

for i, raw in enumerate(samples):
    print(f"--- sample {i} ---")
    print(f"input: {raw!r}")
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        print(f"after strip(backtick): {text!r}")
        if text.lower().startswith("json"):
            text = text[4:].strip()
            print(f"after slice [4:].strip(): {text!r}")
        else:
            print(f"DID NOT START WITH 'json' — first 5 chars: {text[:5]!r}")
    try:
        parsed = json.loads(text)
        print(f"parsed OK: {parsed}")
    except json.JSONDecodeError as e:
        print(f"PARSE FAILED: {e}")
    print()
