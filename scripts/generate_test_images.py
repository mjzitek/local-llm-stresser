"""Generate the bundled test images used by `stresser vision`.

Run once after install (or whenever the images are missing). Produces
deterministic PNGs under files/images/ so the vision benchmark has a known
ground-truth to score against.

Each image has a caption-style description and (optionally) a known piece of
text to extract — both go into a sidecar JSON manifest so the benchmark can
score the model's response.
"""
from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "files" / "images"


def _font(size: int) -> ImageFont.ImageFont:
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]
    for c in candidates:
        if Path(c).exists():
            try:
                return ImageFont.truetype(c, size)
            except OSError:
                pass
    return ImageFont.load_default()


def make_text_block() -> tuple[Image.Image, dict]:
    """A simple sign with two lines of legible text — pure OCR test."""
    img = Image.new("RGB", (640, 360), "white")
    d = ImageDraw.Draw(img)
    d.rectangle([10, 10, 630, 350], outline="black", width=4)
    d.text((40, 80),  "ROOM 247", fill="black", font=_font(72))
    d.text((40, 200), "MEETING IN PROGRESS", fill="darkred", font=_font(36))
    return img, {
        "expected_text": ["ROOM 247", "MEETING IN PROGRESS"],
        "task": "ocr",
        "description": "A sign reading 'ROOM 247' and 'MEETING IN PROGRESS'.",
    }


def make_chart() -> tuple[Image.Image, dict]:
    """Bar chart with labeled values — perception + reading."""
    img = Image.new("RGB", (640, 480), "white")
    d = ImageDraw.Draw(img)
    bars = [("Mon", 12), ("Tue", 18), ("Wed", 7), ("Thu", 22), ("Fri", 15)]
    base_y = 400
    bar_w = 80
    spacing = 30
    start_x = 60
    max_val = max(v for _, v in bars)
    d.text((180, 20), "Tickets Per Day", fill="black", font=_font(28))
    for i, (label, val) in enumerate(bars):
        x = start_x + i * (bar_w + spacing)
        h = int((val / max_val) * 280)
        d.rectangle([x, base_y - h, x + bar_w, base_y], fill="steelblue", outline="black")
        d.text((x + 18, base_y + 10), label, fill="black", font=_font(20))
        d.text((x + 22, base_y - h - 28), str(val), fill="black", font=_font(20))
    return img, {
        "expected_text": ["Mon", "Tue", "Wed", "Thu", "Fri", "Tickets Per Day",
                          "12", "18", "7", "22", "15"],
        "task": "chart",
        "description": "A bar chart titled 'Tickets Per Day' with bars for "
                       "Mon=12, Tue=18, Wed=7, Thu=22, Fri=15. The peak is Thursday.",
        "key_facts": [
            "the chart title is Tickets Per Day",
            "the highest bar is Thursday at 22",
            "the lowest bar is Wednesday at 7",
        ],
    }


def make_scene() -> tuple[Image.Image, dict]:
    """A simple geometric 'scene' — perception, no text."""
    img = Image.new("RGB", (640, 480), "skyblue")
    d = ImageDraw.Draw(img)
    # Ground
    d.rectangle([0, 320, 640, 480], fill="forestgreen")
    # Sun
    d.ellipse([500, 40, 600, 140], fill="yellow", outline="orange", width=3)
    # House
    d.rectangle([180, 220, 400, 360], fill="tan", outline="black", width=2)
    d.polygon([(180, 220), (290, 130), (400, 220)], fill="firebrick", outline="black")
    d.rectangle([260, 280, 320, 360], fill="saddlebrown", outline="black", width=2)
    d.rectangle([200, 240, 240, 280], fill="lightblue", outline="black")
    d.rectangle([340, 240, 380, 280], fill="lightblue", outline="black")
    return img, {
        "expected_text": [],
        "task": "scene",
        "description": "A simple drawing of a tan house with a red triangular "
                       "roof and a brown door, on green grass under a blue sky "
                       "with a yellow sun in the upper right.",
        "key_facts": [
            "there is a house",
            "the roof is red",
            "the sky is blue",
            "the sun is in the upper right",
        ],
    }


def make_code() -> tuple[Image.Image, dict]:
    """Screenshot-style code block — OCR + reasoning."""
    img = Image.new("RGB", (720, 360), "#1e1e1e")
    d = ImageDraw.Draw(img)
    lines = [
        "def fibonacci(n):",
        "    if n <= 1:",
        "        return n",
        "    return fibonacci(n-1) + fibonacci(n-2)",
        "",
        "print(fibonacci(10))",
    ]
    for i, line in enumerate(lines):
        d.text((30, 30 + i * 42), line, fill="#dcdcdc", font=_font(22))
    return img, {
        "expected_text": ["def fibonacci(n):", "fibonacci(10)"],
        "task": "code-ocr",
        "description": "A dark-themed code screenshot showing a Python "
                       "fibonacci function and a call to fibonacci(10).",
        "key_facts": [
            "the language is Python",
            "the function is fibonacci",
            "it is recursive",
        ],
    }


GENERATORS = {
    "text_block.png": make_text_block,
    "chart.png":      make_chart,
    "scene.png":      make_scene,
    "code.png":       make_code,
}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {}
    for name, fn in GENERATORS.items():
        path = OUT / name
        img, meta = fn()
        img.save(path, "PNG", optimize=True)
        manifest[name] = meta
        print(f"  wrote {path.relative_to(ROOT)}  ({path.stat().st_size:,} bytes)")
    (OUT / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"  wrote {(OUT / 'manifest.json').relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
