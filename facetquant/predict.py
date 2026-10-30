import argparse
import csv
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torchvision.transforms.functional import to_tensor

from .model import load_model


EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def image_paths(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    return sorted(p for p in path.rglob("*") if p.suffix.lower() in EXTENSIONS)


def draw_histogram(areas: list[float], unit: str, output: Path) -> None:
    width, height = 1100, 650
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font_path = "C:/Windows/Fonts/arial.ttf" if Path("C:/Windows/Fonts/arial.ttf").exists() else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 22) if Path(font_path).exists() else ImageFont.load_default()
    draw.text((70, 25), f"Crystal area distribution (n={len(areas)})", fill="black", font=font)
    draw.line((80, 570, 1050, 570), fill="black", width=2)
    draw.line((80, 75, 80, 570), fill="black", width=2)
    draw.text((80, 595), f"Area ({unit})", fill="black", font=font)
    draw.text((15, 75), "Count", fill="black", font=font)
    if areas:
        bins = min(30, max(5, round(math.sqrt(len(areas)))))
        counts, edges = np.histogram(areas, bins=bins)
        peak = max(1, int(counts.max()))
        for i, count in enumerate(counts):
            x0 = 82 + round(i * 960 / bins)
            x1 = 82 + round((i + 1) * 960 / bins) - 2
            y0 = 568 - round(int(count) * 460 / peak)
            draw.rectangle((x0, y0, x1, 568), fill="#3178c6")
        draw.text((82, 575), f"{edges[0]:.1f}", fill="black", font=font)
        draw.text((925, 575), f"{edges[-1]:.1f}", fill="black", font=font)
        draw.text((25, 80), str(peak), fill="black", font=font)
    image.save(output)


@torch.inference_mode()
def predict(args):
    source = Path(args.input).expanduser().resolve()
    paths = image_paths(source)
    if not paths:
        raise FileNotFoundError(f"No images found at {source}")
    out = Path(args.out).resolve()
    (out / "overlays").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    model = load_model(args.weights, args.device, args.max_size)
    rows = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        prediction = model([to_tensor(image).to(args.device)])[0]
        scores = prediction["scores"].cpu().numpy()
        keep = np.flatnonzero(scores >= args.score_threshold)
        masks = prediction["masks"][keep, 0].cpu().numpy() >= args.mask_threshold
        boxes = prediction["boxes"][keep].cpu().numpy()
        scores = scores[keep]
        if args.exclude_border and len(masks):
            interior = [i for i, mask in enumerate(masks) if not (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any())]
            masks, boxes, scores = masks[interior], boxes[interior], scores[interior]
        overlay = np.asarray(image).astype(np.float32).copy()
        relative = path.relative_to(source) if source.is_dir() else Path(path.name)
        stem = relative.with_suffix("")
        (out / "overlays" / stem.parent).mkdir(parents=True, exist_ok=True)
        (out / "masks" / stem.parent).mkdir(parents=True, exist_ok=True)
        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores), 1):
            area = int(mask.sum())
            hue = (i * 0.61803398875) % 1
            import colorsys
            color = np.array(colorsys.hsv_to_rgb(hue, 0.85, 1.0)) * 255
            overlay[mask] = overlay[mask] * 0.4 + color * 0.6
            row = {"image": str(relative).replace("\\", "/"), "crystal_id": i, "score": round(float(score), 5), "area_px2": area, "x_min": round(float(box[0]), 1), "y_min": round(float(box[1]), 1), "x_max": round(float(box[2]), 1), "y_max": round(float(box[3]), 1)}
            if args.um_per_pixel is not None:
                row["area_um2"] = round(area * args.um_per_pixel**2, 4)
            rows.append(row)
        Image.fromarray(overlay.clip(0, 255).astype(np.uint8)).save(out / "overlays" / stem.with_suffix(".png"))
        np.savez_compressed(out / "masks" / stem.with_suffix(".npz"), masks=masks, scores=scores)
        print(f"{relative}: {len(masks)} crystals", flush=True)
    fields = ["image", "crystal_id", "score", "area_px2"]
    if args.um_per_pixel is not None:
        fields.append("area_um2")
    fields += ["x_min", "y_min", "x_max", "y_max"]
    with (out / "crystals.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    unit = "um^2" if args.um_per_pixel is not None else "px^2"
    key = "area_um2" if args.um_per_pixel is not None else "area_px2"
    draw_histogram([float(row[key]) for row in rows], unit, out / "area_distribution.png")
    print(f"Saved {len(rows)} crystal measurements to {out}", flush=True)


def add_predict_arguments(parser: argparse.ArgumentParser):
    parser.add_argument("--input", required=True, help="Single image or image directory")
    parser.add_argument("--weights", required=True, help="FacetQuant FP16 or FP32 state_dict")
    parser.add_argument("--out", default="runs/predictions")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--max-size", type=int, default=1024)
    parser.add_argument("--score-threshold", type=float, default=0.6)
    parser.add_argument("--mask-threshold", type=float, default=0.5)
    parser.add_argument("--um-per-pixel", type=float, help="Micrometres per pixel in the ORIGINAL image")
    parser.add_argument("--exclude-border", action="store_true", help="Exclude crystals touching the image boundary")
