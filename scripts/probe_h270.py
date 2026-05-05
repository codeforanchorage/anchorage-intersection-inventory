"""One-off diagnostic: replay h270 of osm_31329116 and print Claude's raw response."""

import base64
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
from anthropic import Anthropic

import config
from src.detect_assets import FULL_IMAGE_PROMPT

load_dotenv(config.PROJECT_ROOT / ".env")
client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

img_path = config.IMAGE_DIR / "osm_31329116_h270.jpg"
b64 = base64.standard_b64encode(img_path.read_bytes()).decode("ascii")

response = client.messages.create(
    model="claude-sonnet-4-6",
    max_tokens=2048,
    messages=[{
        "role": "user",
        "content": [
            {"type": "image", "source": {
                "type": "base64", "media_type": "image/jpeg", "data": b64,
            }},
            {"type": "text", "text": FULL_IMAGE_PROMPT},
        ],
    }],
)

raw = "".join(b.text for b in response.content if getattr(b, "type", "") == "text")
print("=" * 60)
print("RAW RESPONSE:")
print("=" * 60)
print(raw)
print("=" * 60)
print(f"length: {len(raw)}  stop_reason: {response.stop_reason}")
