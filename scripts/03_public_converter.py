"""Script #3 — Public Dataset Surgical Converter (IDD Kaggle & Mapillary Vistas v2 HF) -> 11-Class YOLO-seg.

Key data-centric features:
  1. Rare-Class Priority Selection: prioritizes scenes rich in rare competition classes
     (Other vehicle, Cyclist, Bicycle, Bus, Traffic light, Sign) instead of car-only spam.
  2. Explicit Rider-Vehicle Polygon Union: merges `rider`/`motorcyclist`/`bicyclist` polygons
     with their underlying `motorcycle`/`bicycle` into ONE single mask (Motorcyclist=5, Cyclist=7).
  3. Z-Order Non-Overlap + <64 px²@640 filtering + Resize max-width 1280px.
  4. Automatic Provenance Report (`public_provenance.csv` + `public_provenance_summary.md`)
     ready to copy directly into Slide #2 (50% Presentation Score)!

Usage examples:
  # Convert IDD directly on Kaggle (/kaggle/input/indian-driving-dataset — 0s download!)
  python scripts/03_public_converter.py \
      --source idd \
      --input /kaggle/input/indian-driving-dataset \
      --out data/processed/public_idd \
      --max-images 600

  # Stream Mapillary Vistas v2 directly from HuggingFace (no 20GB upfront download!)
  python scripts/03_public_converter.py \
      --source mapillary_hf \
      --out data/processed/public_mapillary \
      --max-images 1200
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from common.seg_utils import (
    CLASS_DISPLAY_VI,
    CLASSES,
    MIN_GT_AREA_PX,
    clean_and_normalize_polygon,
    enforce_non_overlap,
    merge_riders_and_vehicles,
    write_data_yaml,
    write_yolo_seg_txt,
)

IDD_MAPPING: dict[str, int | str] = {
    "car": 0,
    "bus": 1,
    "truck": 2,
    "autorickshaw": 3,
    "caravan": 3,
    "trailer": 3,
    "vehicle fallback": 3,
    "motorcycle": 4,
    "bicycle": 6,
    "person": 8,
    "rider": "rider",
    "traffic sign": 9,
    "traffic light": 10,
}

MAPILLARY_MAPPING: dict[str, int | str] = {
    "object--vehicle--car": 0,
    "object--vehicle--bus": 1,
    "object--vehicle--truck": 2,
    "object--vehicle--trailer": 3,
    "object--vehicle--wheeled-slow": 3,
    "object--vehicle--other-vehicle": 3,
    "object--vehicle--caravan": 3,
    "object--vehicle--motorcycle": 4,
    "object--vehicle--bicycle": 6,
    "human--person": 8,
    "human--person--individual": 8,
    "human--person--person-group": 8,
    "human--rider--motorcyclist": "motorcyclist_rider",
    "human--rider--bicyclist": "bicyclist_rider",
    "human--rider--other-rider": "rider",
    "object--traffic-sign--front": 9,
    "object--traffic-sign--back": 9,
    "object--traffic-sign--information-parking": 9,
    "object--traffic-sign--temporary-front": 9,
    "object--traffic-sign--temporary-back": 9,
    "object--traffic-sign--direction-front": 9,
    "object--traffic-sign--direction-back": 9,
    "object--traffic-sign--ambiguous": 9,
    "object--traffic-light": 10,
    "object--traffic-light--general-upright": 10,
    "object--traffic-light--general-horizontal": 10,
    "object--traffic-light--general-single": 10,
    "object--traffic-light--pedestrians": 10,
    "object--traffic-light--cyclists": 10,
    "object--traffic-light--other": 10,
}

RARE_WEIGHTS: dict[int, float] = {
    0: 0.5,
    1: 3.0,
    2: 2.0,
    3: 6.0,   # Other vehicle
    4: 3.0,   # Motorcycle
    5: 2.5,   # Motorcyclist
    6: 5.0,   # Bicycle
    7: 6.0,   # Cyclist
    8: 1.0,   # Pedestrian
    9: 3.5,   # Sign
    10: 4.5,  # Traffic light
}


def resize_pil_max_width(img: Image.Image, max_width: int) -> Image.Image:
    w, h = img.size
    if max_width > 0 and w > max_width:
        scale = max_width / float(w)
        return img.resize((max_width, max(1, int(round(h * scale)))), Image.Resampling.LANCZOS)
    return img


def compute_priority_score(final_items: list[dict[str, Any]]) -> float:
    if not final_items:
        return 0.0
    present = {int(it["class_id"]) for it in final_items}
    base = sum(RARE_WEIGHTS.get(c, 1.0) for c in present)
    rare_cnt = sum(1 for c in present if c in (1, 3, 4, 6, 7, 9, 10))
    return base + 2.0 * rare_cnt + 0.05 * min(len(final_items), 30)


def find_idd_pairs(root: Path) -> list[tuple[Path, Path]]:
    """Locate matching (image_path, *_polygons.json) pairs inside any Kaggle IDD layout."""
    json_files = sorted(root.rglob("*_polygons.json"))
    if not json_files:
        json_files = sorted(p for p in root.rglob("*.json") if "polygon" in p.name.lower())
    if not json_files:
        raise FileNotFoundError(f"No IDD polygon JSON files found under {root}")

    img_index: dict[str, Path] = {}
    for ext in ("*.png", "*.jpg", "*.jpeg"):
        for p in root.rglob(ext):
            low = str(p).lower()
            if "gtfine" in low or "mask" in p.name.lower() or "label" in p.name.lower():
                continue
            stem = p.stem
            img_index[stem] = p
            for suffix in ("_leftImg8bit", "_image"):
                if stem.endswith(suffix):
                    img_index[stem[: -len(suffix)]] = p

    pairs: list[tuple[Path, Path]] = []
    for jf in json_files:
        base = jf.stem
        for suffix in ("_gtFine_polygons", "_polygons"):
            if base.endswith(suffix):
                base = base[: -len(suffix)]
                break
        img_p = img_index.get(base) or img_index.get(f"{base}_leftImg8bit")
        if img_p is not None and img_p.is_file():
            pairs.append((img_p, jf))
    return pairs


def convert_idd(
    input_dir: Path,
    out_dir: Path,
    max_images: int,
    max_width: int = 1280,
    require_sign_or_light: bool = False,
) -> dict[str, Any]:
    pairs = find_idd_pairs(input_dir)
    print(f"[IDD] Found {len(pairs)} matched (image, polygon JSON) pairs under {input_dir}")

    scored: list[tuple[float, Path, list[dict[str, Any]]]] = []
    for img_p, json_p in pairs:
        try:
            data = json.loads(json_p.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            continue
        w = int(data.get("imgWidth") or 0)
        h = int(data.get("imgHeight") or 0)
        if w <= 0 or h <= 0:
            with Image.open(img_p) as im:
                w, h = im.size

        raw_insts: list[dict[str, Any]] = []
        for obj in data.get("objects", []):
            lbl = str(obj.get("label", "")).strip().lower()
            if lbl not in IDD_MAPPING:
                continue
            pts = np.asarray(obj.get("polygon", []), dtype=np.float32)
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
                continue
            pts[:, 0] /= float(w)
            pts[:, 1] /= float(h)
            cleaned = clean_and_normalize_polygon(pts, w, h, min_area_640=MIN_GT_AREA_PX)
            if cleaned is not None:
                raw_insts.append({"role": IDD_MAPPING[lbl], "poly": cleaned, "score": 1.0})

        if not raw_insts:
            continue
        merged = merge_riders_and_vehicles(raw_insts, mode="explicit_rider", w=w, h=h, min_area_640=MIN_GT_AREA_PX)
        final_items = enforce_non_overlap(merged, w=w, h=h, min_area_640=MIN_GT_AREA_PX)
        if not final_items:
            continue
        if require_sign_or_light and not any(it["class_id"] in (9, 10) for it in final_items):
            continue
        scored.append((compute_priority_score(final_items), img_p, final_items))

    scored.sort(key=lambda t: t[0], reverse=True)
    selected = scored[:max_images] if max_images > 0 else scored

    img_out = out_dir / "images"
    lab_out = out_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)

    cls_counts: Counter[int] = Counter()
    prov_rows: list[dict[str, Any]] = []
    for idx, (score, img_p, items) in enumerate(selected):
        fname = f"idd_{idx:05d}_{img_p.stem}.jpg"
        with Image.open(img_p) as im:
            resize_pil_max_width(im.convert("RGB"), max_width).save(img_out / fname, quality=92)
        write_yolo_seg_txt(lab_out / f"{Path(fname).stem}.txt", items)
        for it in items:
            cls_counts[int(it["class_id"])] += 1
        prov_rows.append({"file": fname, "source_file": img_p.name, "priority_score": round(score, 2), "n_instances": len(items)})

    return {
        "dataset": "Indian Driving Dataset (IDD)",
        "url": "https://www.kaggle.com/datasets/manjotpahwa/indian-driving-dataset",
        "license": "IIIT Hyderabad Academic / Non-Commercial Research",
        "scanned": len(pairs),
        "kept": len(selected),
        "per_class_instances": {CLASSES[i]: cls_counts[i] for i in range(len(CLASSES))},
        "rows": prov_rows,
    }


def _extract_mapillary_instances(
    inst_arr: np.ndarray,
    label_names: list[str],
    w: int,
    h: int,
) -> list[dict[str, Any]]:
    try:
        import cv2
    except ImportError as exc:
        raise SystemExit("Missing opencv-python-headless. Install with: pip install opencv-python-headless") from exc

    raw_insts: list[dict[str, Any]] = []
    for val in np.unique(inst_arr):
        cls_idx = int(val) // 256
        if not (0 <= cls_idx < len(label_names)):
            continue
        lname = label_names[cls_idx]
        if lname not in MAPILLARY_MAPPING:
            continue
        u8 = (inst_arr == val).astype(np.uint8) * 255
        contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            continue
        largest = max(contours, key=cv2.contourArea)
        if len(largest) < 3:
            continue
        eps = 0.002 * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, eps, True).reshape(-1, 2).astype(np.float32)
        if len(approx) < 3:
            approx = largest.reshape(-1, 2).astype(np.float32)
        approx[:, 0] /= float(w)
        approx[:, 1] /= float(h)
        cleaned = clean_and_normalize_polygon(approx, w, h, min_area_640=MIN_GT_AREA_PX)
        if cleaned is not None:
            raw_insts.append({"role": MAPILLARY_MAPPING[lname], "poly": cleaned, "score": 1.0})
    return raw_insts


def convert_mapillary_hf(
    out_dir: Path,
    max_images: int,
    max_scan: int = 4000,
    min_priority: float = 10.0,
    max_width: int = 1280,
    hf_repo: str = "candylion/mapillary-vistas-v2",
) -> dict[str, Any]:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit("Missing `datasets`. Install with: pip install datasets huggingface-hub") from exc

    print(f"[Mapillary HF] Streaming {hf_repo} (target={max_images}, max_scan={max_scan})...")
    ds = load_dataset(hf_repo, split="train", streaming=True)

    img_out = out_dir / "images"
    lab_out = out_dir / "labels"
    img_out.mkdir(parents=True, exist_ok=True)
    lab_out.mkdir(parents=True, exist_ok=True)

    cls_counts: Counter[int] = Counter()
    prov_rows: list[dict[str, Any]] = []
    kept = 0
    scanned = 0
    label_names: list[str] | None = None

    for idx, sample in enumerate(ds):
        if kept >= max_images or scanned >= max_scan:
            break
        scanned += 1
        img = sample.get("image")
        inst_mask = sample.get("instance") or sample.get("instances") or sample.get("label") or sample.get("panoptic")
        if img is None or inst_mask is None:
            if scanned == 1:
                raise RuntimeError(f"Unexpected HF dataset schema! Available keys: {list(sample.keys())}")
            continue

        if label_names is None:
            if "label_names" in sample and isinstance(sample["label_names"], list):
                label_names = sample["label_names"]
            else:
                feat = getattr(ds, "features", None)
                if feat and "label" in feat and hasattr(feat["label"], "names"):
                    label_names = list(feat["label"].names)
                else:
                    raise RuntimeError(
                        f"Could not auto-detect Mapillary label names from `{hf_repo}`. "
                        f"Available keys={list(sample.keys())}."
                    )

        rgb = img.convert("RGB") if isinstance(img, Image.Image) else Image.fromarray(np.asarray(img)).convert("RGB")
        w, h = rgb.size
        inst_arr = np.asarray(inst_mask)
        if inst_arr.ndim != 2:
            if scanned == 1:
                raise RuntimeError(f"Expected 2D uint16/int32 instance mask, got shape={inst_arr.shape}, dtype={inst_arr.dtype}")
            continue

        raw_insts = _extract_mapillary_instances(inst_arr, label_names, w, h)
        if not raw_insts:
            continue
        merged = merge_riders_and_vehicles(raw_insts, mode="explicit_rider", w=w, h=h, min_area_640=MIN_GT_AREA_PX)
        final_items = enforce_non_overlap(merged, w=w, h=h, min_area_640=MIN_GT_AREA_PX)
        score = compute_priority_score(final_items)
        if score < min_priority:
            continue

        fname = f"map_{idx:06d}.jpg"
        resize_pil_max_width(rgb, max_width).save(img_out / fname, quality=92)
        write_yolo_seg_txt(lab_out / f"{Path(fname).stem}.txt", final_items)
        for it in final_items:
            cls_counts[int(it["class_id"])] += 1
        prov_rows.append({"file": fname, "source_file": f"hf_idx_{idx}", "priority_score": round(score, 2), "n_instances": len(final_items)})
        kept += 1
        if kept % 100 == 0:
            print(f"  [Mapillary HF] kept={kept}/{max_images} (scanned={scanned})")

    return {
        "dataset": "Mapillary Vistas v2",
        "url": f"https://huggingface.co/datasets/{hf_repo}",
        "license": "Mapillary Vistas Research License (Non-Commercial / Academic)",
        "scanned": scanned,
        "kept": kept,
        "per_class_instances": {CLASSES[i]: cls_counts[i] for i in range(len(CLASSES))},
        "rows": prov_rows,
    }


def save_provenance(out_dir: Path, summary: dict[str, Any]) -> None:
    rows = summary.pop("rows", [])
    if rows:
        with open(out_dir / "public_provenance.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    (out_dir / "public_provenance_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md = [
        f"# Public Dataset Provenance — {summary['dataset']}",
        f"- **URL:** `{summary['url']}`",
        f"- **License:** {summary['license']}",
        f"- **Scanned:** `{summary['scanned']}` | **Kept:** `{summary['kept']}`",
        "",
        "| ID | Lớp | Tiếng Việt | Instances |",
        "|---:|---|---|---:|",
    ]
    for i, cname in enumerate(CLASSES):
        md.append(f"| {i} | `{cname}` | {CLASS_DISPLAY_VI[cname]} | {summary['per_class_instances'].get(cname, 0)} |")
    (out_dir / "public_provenance_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    write_data_yaml(out_dir / "data.yaml", train_rel="images", val_rel="images")
    print("\n".join(md))


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Convert IDD / Mapillary Vistas v2 into 11-class YOLO-seg.")
    ap.add_argument("--source", choices=["idd", "idd_kagglehub", "mapillary_hf"], required=True)
    ap.add_argument("--input", type=Path, default=None, help="Input directory (required for --source idd)")
    ap.add_argument("--kaggle-slug", type=str, default="manjotpahwa/indian-driving-dataset")
    ap.add_argument("--out", type=Path, required=True, help="Output directory")
    ap.add_argument("--max-images", type=int, default=800)
    ap.add_argument("--max-scan", type=int, default=4000)
    ap.add_argument("--min-priority", type=float, default=10.0)
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--require-sign-or-light", action="store_true")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.source == "idd":
        if args.input is None:
            raise SystemExit("--input is required when --source idd")
        summary = convert_idd(args.input, args.out, args.max_images, args.max_width, args.require_sign_or_light)
    elif args.source == "idd_kagglehub":
        import kagglehub

        print(f"[IDD] Downloading public Kaggle dataset `{args.kaggle_slug}` via kagglehub...")
        dl_dir = Path(kagglehub.dataset_download(args.kaggle_slug))
        print(f"[IDD] Downloaded to: {dl_dir}")
        summary = convert_idd(dl_dir, args.out, args.max_images, args.max_width, args.require_sign_or_light)
    else:
        summary = convert_mapillary_hf(args.out, args.max_images, args.max_scan, args.min_priority, args.max_width)
    save_provenance(args.out, summary)


if __name__ == "__main__":
    main()
