"""
One-time script: stitches query + top-5 results into a side-by-side grid image.
Run from project root: python notebooks/make_grid.py
"""
import os
from PIL import Image, ImageDraw, ImageFont

THUMB = 150       # thumbnail size
PAD   = 6         # padding between images
LABEL_H = 18      # height reserved for label text above each image

def make_grid(folder, out_path):
    """
    folder: e.g. notebooks/checks/finetuned_full/Dresses_39716
    out_path: where to save the stitched image
    """
    query_img = Image.open(os.path.join(folder, "query.jpg")).resize((THUMB, THUMB))
    results = []
    for i in range(1, 6):
        # result files are named result_1.jpg, result_2.jpg, etc.
        candidates = [f for f in os.listdir(folder) if f.startswith(f"result_{i}")]
        if candidates:
            results.append(Image.open(os.path.join(folder, candidates[0])).resize((THUMB, THUMB)))

    images = [query_img] + results
    labels = ["Query"] + [f"Result {i}" for i in range(1, len(results) + 1)]

    n = len(images)
    W = n * THUMB + (n - 1) * PAD
    H = THUMB + LABEL_H + PAD

    canvas = Image.new("RGB", (W, H), color=(245, 245, 245))
    draw = ImageDraw.Draw(canvas)

    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 12)
    except Exception:
        font = ImageFont.load_default()

    for i, (img, label) in enumerate(zip(images, labels)):
        x = i * (THUMB + PAD)
        draw.text((x + 4, 2), label, fill=(80, 80, 80), font=font)
        canvas.paste(img, (x, LABEL_H))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    canvas.save(out_path)
    print(f"saved: {out_path}")


if __name__ == "__main__":
    # dresses before/after — the hero comparison
    make_grid(
        "notebooks/checks/baseline_full/Dresses_39716",
        "notebooks/checks/grids/dresses_baseline.jpg"
    )
    make_grid(
        "notebooks/checks/finetuned_full/Dresses_39716",
        "notebooks/checks/grids/dresses_finetuned.jpg"
    )

    # optional: generate grids for all 5 query classes, both models
    for model_prefix in ["baseline_full", "finetuned_full"]:
        checks_dir = f"notebooks/checks/{model_prefix}"
        if not os.path.exists(checks_dir):
            continue
        for folder_name in os.listdir(checks_dir):
            folder = os.path.join(checks_dir, folder_name)
            if os.path.isdir(folder):
                out = f"notebooks/checks/grids/{model_prefix}_{folder_name}.jpg"
                make_grid(folder, out)