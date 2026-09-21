"""Generate N synthetic 448x448 images (3 shapes, random quadrants/colours) + truth.json for fresh_bench.py.

Usage: python scripts/make_shapes.py OUT_DIR [N] [SEED]
"""

import json
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

QUADS = {"top_left": (0, 0), "top_right": (224, 0), "bottom_left": (0, 224), "bottom_right": (224, 224)}
COLOURS = {"red": (220, 30, 30), "blue": (30, 60, 220), "green": (30, 170, 60), "yellow": (230, 200, 30)}


def main():
    out = Path(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    random.seed(int(sys.argv[3]) if len(sys.argv) > 3 else 7)
    out.mkdir(parents=True, exist_ok=True)
    truth = {}
    for i in range(n):
        img = Image.new("RGB", (448, 448), "white")
        draw = ImageDraw.Draw(img)
        quads = random.sample(list(QUADS), 3)
        shapes = ["circle", "square", "triangle"]
        random.shuffle(shapes)
        colours = random.sample(list(COLOURS), 3)
        placed = {}
        for shape, quad, colour in zip(shapes, quads, colours, strict=True):
            x0, y0 = QUADS[quad]
            x0, y0 = x0 + 30, y0 + 30
            x1, y1 = x0 + 160, y0 + 160
            if shape == "circle":
                draw.ellipse((x0, y0, x1, y1), fill=COLOURS[colour])
            elif shape == "square":
                draw.rectangle((x0, y0, x1, y1), fill=COLOURS[colour])
            else:
                draw.polygon([((x0 + x1) // 2, y0), (x0, y1), (x1, y1)], fill=COLOURS[colour])
            placed[shape] = {"quadrant": quad, "colour": colour}
        truth[f"img{i}.png"] = {
            "shapes": placed,
            "red_shape": next((s for s, v in placed.items() if v["colour"] == "red"), "none"),
            "count": "3",
            "blue_square": placed["square"]["colour"] == "blue",
            "circle_quadrant": placed["circle"]["quadrant"],
        }
        img.save(out / f"img{i}.png")
    (out / "truth.json").write_text(json.dumps(truth, indent=1))
    print(f"wrote {n} images to {out}")


if __name__ == "__main__":
    main()
