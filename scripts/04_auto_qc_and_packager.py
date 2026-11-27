#!/usr/bin/env python3
"""Script #4: Automated Data QC Gate & Competition Zip Packager.

Run BEFORE every training iteration (V1 -> V4) to guarantee:
  1. Zero filename collisions across merged sources (prefix per source directory)
  2. Valid YOLO-seg polygons (clipped [0, 1], shapely-repaired, >= 3 vertices)
  3. Zero tiny GT noise (< 64 px² at letterboxed 640 scale)
  4. Non-overlapping visible masks (optional --enforce-non-overlap, default ON)
  5. Zero duplicate / Train-Val leakage via 64-bit dHash (Golden Val is always protected!)
  6. Full Markdown QC Report (ready for Slide) + visual contact-sheets + dataset_vX.zip

Usage:
    python scripts/04_auto_qc_and_packager.py \
        --train-dirs data/processed/public_mapillary data/processed/public_idd data/processed/cvat_vn_train \
        --val-dir data/processed/cvat_golden_val \
        --hard-neg-dir data/raw/vn_hard_negatives \
        --out-zip data/processed/dataset_v1.zip
"""
from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import sys
import zipfile
from collections import Counter
from pathlib import Path

from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.seg_utils import (  # noqa: E402
    CLASSES,
    CLASS_DISPLAY_VI,
    IMG_EXTS,
    MIN_GT_AREA_PX,
    Instance,
    clean_polygon,
    compute_dhash,
    enforce_non_overlap,
    hamming_distance,
    render_contact_sheet,
    read_yolo_seg_file,
    write_data_yaml,
    write_yolo_seg_file,
)


def _discover_images_and_labels(src_dir: Path) -> list[tuple[Path, Path | None]]:
    """Support both flat (images/ + labels/) and nested (images/train + labels/train) layouts."""
    candidates = [
        (src_dir / "images" / "train", src_dir / "labels" / "train"),
        (src_dir / "images", src_dir / "labels"),
        (src_dir, src_dir),
    ]
    for img_dir, lbl_dir in candidates:
        if img_dir.is_dir():
            imgs = sorted(p for p in img_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
            if imgs:
                return [(p, (lbl_dir / f"{p.stem}.txt") if lbl_dir.is_dir() else None) for p in imgs]
    return []


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def process_split_items(
    sources: list[tuple[str, list[tuple[Path, Path | None]], bool]],
    out_img_dir: Path,
    out_lbl_dir: Path,
    enforce_overlap: bool,
    min_area_640: float,
    dup_hamming_max: int,
    forbidden_hashes: list[int],
    leak_hamming_max: int,
    max_width: int,
) -> dict:
    """Clean, deduplicate, QC, and write one split (val or train)."""
    out_img_dir.mkdir(parents=True, exist_ok=True)
    out_lbl_dir.mkdir(parents=True, exist_ok=True)

    seen_hashes: list[int] = []
    cls_instances = Counter()
    cls_images = Counter()
    stats = {
        "images": 0,
        "empty_images": 0,
        "instances_before": 0,
        "instances_after": 0,
        "dropped_tiny_or_invalid": 0,
        "dropped_dup_self": 0,
        "dropped_leak_val": 0,
        "parse_errors": [],
        "saved_pairs": [],
        "hashes": seen_hashes,
    }

    for prefix, pairs, is_hard_neg in sources:
        for img_path, lbl_path in pairs:
            try:
                with Image.open(img_path) as im:
                    im = ImageOps.exif_transpose(im).convert("RGB")
                    if max_width and im.width > max_width:
                        new_h = int(round(im.height * (max_width / im.width)))
                        im = im.resize((max_width, new_h), Image.Resampling.LANCZOS)
                    w, h = im.size
                    dh = compute_dhash(im)

                    if forbidden_hashes and any(hamming_distance(dh, fh) <= leak_hamming_max for fh in forbidden_hashes):
                        stats["dropped_leak_val"] += 1
                        continue
                    if dup_hamming_max >= 0 and any(hamming_distance(dh, sh) <= dup_hamming_max for sh in seen_hashes):
                        stats["dropped_dup_self"] += 1
                        continue

                    if is_hard_neg or lbl_path is None or not lbl_path.is_file():
                        raw_insts, errs = [], []
                    else:
                        raw_insts, errs = read_yolo_seg_file(lbl_path)
                        stats["parse_errors"].extend(errs)

                    stats["instances_before"] += len(raw_insts)

                    cleaned_insts: list[Instance] = []
                    for inst in raw_insts:
                        c_poly = clean_polygon(inst.polygon, w, h, min_area_640=min_area_640)
                        if c_poly is not None:
                            cleaned_insts.append(Instance(class_id=inst.class_id, polygon=c_poly, score=inst.score))
                        else:
                            stats["dropped_tiny_or_invalid"] += 1

                    if enforce_overlap and len(cleaned_insts) > 1:
                        final_insts = enforce_non_overlap(cleaned_insts, w, h, min_area_640=min_area_640)
                    else:
                        final_insts = cleaned_insts

                    stem = f"{prefix}_{img_path.stem}"
                    dst_img = out_img_dir / f"{stem}.jpg"
                    dst_lbl = out_lbl_dir / f"{stem}.txt"
                    im.save(dst_img, quality=92)
                    write_yolo_seg_file(dst_lbl, final_insts)

                    seen_hashes.append(dh)
                    stats["images"] += 1
                    stats["instances_after"] += len(final_insts)
                    if not final_insts:
                        stats["empty_images"] += 1
                    for c in {i.class_id for i in final_insts}:
                        cls_images[c] += 1
                    for i in final_insts:
                        cls_instances[i.class_id] += 1
                    stats["saved_pairs"].append((dst_img, final_insts))
            except Exception as exc:
                stats["parse_errors"].append(f"{img_path.name}: failed to process ({exc})")

    stats["cls_instances"] = cls_instances
    stats["cls_images"] = cls_images
    return stats


def build_qc_report(train_stats: dict, val_stats: dict, out_md: Path, zip_path: Path) -> str:
    t_counts = [train_stats["cls_instances"].get(i, 0) for i in range(len(CLASSES))]
    nonzero = [c for c in t_counts if c > 0]
    imbalance = (max(nonzero) / min(nonzero)) if len(nonzero) >= 2 else float("inf")

    lines = [
        "# Auto-QC Dataset Report (Đội 18 · `TEAM_ID = 23`)",
        "",
        f"- **Output Zip:** `{zip_path.name}` ({zip_path.stat().st_size / (1024 * 1024):.2f} MiB)",
        f"- **Zip SHA-256:** `{_sha256_file(zip_path)}`",
        f"- **Train:** {train_stats['images']} ảnh ({train_stats['empty_images']} ảnh nền/Hard Neg) | {train_stats['instances_after']} instances",
        f"- **Val (Golden VN):** {val_stats['images']} ảnh ({val_stats['empty_images']} ảnh nền) | {val_stats['instances_after']} instances",
        f"- **Imbalance Ratio (Train max/min):** `{imbalance:.1f}x` *(BTC cảnh báo > 30x; mục tiêu Top 1 ≤ 15x)*",
        f"- **Auto-QC Cleaned:** bỏ `{train_stats['dropped_tiny_or_invalid']}` mask lỗi/<64px²; chặn `{train_stats['dropped_leak_val']}` ảnh train rò rỉ với Val; loại `{train_stats['dropped_dup_self']}` ảnh trùng nội bộ.",
        "",
        "| ID | Class | Tiếng Việt | Train Inst | Train Img | Val Inst | Val Img | Trạng thái |",
        "|---:|---|---|---:|---:|---:|---:|---|",
    ]
    for cid, cname in enumerate(CLASSES):
        ti = train_stats["cls_instances"].get(cid, 0)
        timg = train_stats["cls_images"].get(cid, 0)
        vi = val_stats["cls_instances"].get(cid, 0)
        vimg = val_stats["cls_images"].get(cid, 0)
        flags = []
        if ti < 100:
            flags.append("THIẾU TRAIN (<100)")
        elif ti < 300:
            flags.append("Nên bổ sung (<300)")
        if vi < 10:
            flags.append("THIẾU VAL (<10)")
        status = ", ".join(flags) if flags else "OK"
        lines.append(
            f"| {cid} | `{cname}` | {CLASS_DISPLAY_VI[cname]} | {ti} | {timg} | {vi} | {vimg} | {status} |"
        )

    if train_stats["parse_errors"] or val_stats["parse_errors"]:
        errs = (train_stats["parse_errors"] + val_stats["parse_errors"])[:15]
        lines.extend(["", "## Cảnh báo lỗi file gốc (tối đa 15 dòng)", ""])
        lines.extend(f"- {e}" for e in errs)

    text = "\n".join(lines) + "\n"
    out_md.write_text(text, encoding="utf-8")
    return text


def package_dataset_zip(stage_dir: Path, out_zip: Path) -> Path:
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out_zip, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(stage_dir.rglob("*")):
            if p.is_file():
                zf.write(p, arcname=str(p.relative_to(stage_dir)).replace("\\", "/"))
    return out_zip


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Auto-QC, deduplicate, enforce non-overlap, and package dataset_vX.zip.")
    ap.add_argument("--train-dirs", type=Path, nargs="+", required=True, help="One or more train source dirs")
    ap.add_argument("--val-dir", type=Path, required=True, help="Golden VN Val source dir")
    ap.add_argument("--hard-neg-dir", type=Path, default=None, help="Optional dir of background/hard-negative images")
    ap.add_argument("--out-zip", type=Path, required=True, help="Output competition dataset zip")
    ap.add_argument("--work-dir", type=Path, default=None, help="Staging dir (default: alongside out-zip)")
    ap.add_argument("--no-non-overlap", action="store_true", help="Disable non-overlap subtraction")
    ap.add_argument("--min-area-640", type=float, default=MIN_GT_AREA_PX)
    ap.add_argument("--dup-hamming-max", type=int, default=2, help="Deduplicate near-identical train images")
    ap.add_argument("--leak-hamming-max", type=int, default=4, help="Block train images close to any val image")
    ap.add_argument("--max-width", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=23)
    args = ap.parse_args()

    stage = args.work_dir or (args.out_zip.parent / f"_{args.out_zip.stem}_stage")
    if stage.exists():
        shutil.rmtree(stage)
    stage.mkdir(parents=True)

    # 1. Process Golden Val FIRST (protected from leakage)
    val_pairs = _discover_images_and_labels(args.val_dir)
    if not val_pairs:
        raise SystemExit(f"No images found in --val-dir: {args.val_dir}")
    print(f"[qc] Processing Golden Val ({len(val_pairs)} images) from {args.val_dir} ...")
    val_stats = process_split_items(
        sources=[("val", val_pairs, False)],
        out_img_dir=stage / "images" / "val",
        out_lbl_dir=stage / "labels" / "val",
        enforce_overlap=not args.no_non_overlap,
        min_area_640=args.min_area_640,
        dup_hamming_max=args.dup_hamming_max,
        forbidden_hashes=[],
        leak_hamming_max=-1,
        max_width=args.max_width,
    )

    # 2. Process Train sources + optional Hard Negatives (cross-checked against Val hashes!)
    train_sources = []
    for idx, d in enumerate(args.train_dirs):
        pairs = _discover_images_and_labels(d)
        prefix = f"s{idx}_{d.name[:12]}"
        print(f"[qc] Discovered {len(pairs)} train images in {d} (prefix={prefix})")
        train_sources.append((prefix, pairs, False))
    if args.hard_neg_dir is not None:
        hn_pairs = _discover_images_and_labels(args.hard_neg_dir)
        print(f"[qc] Discovered {len(hn_pairs)} hard-negative background images in {args.hard_neg_dir}")
        train_sources.append(("hn", hn_pairs, True))

    train_stats = process_split_items(
        sources=train_sources,
        out_img_dir=stage / "images" / "train",
        out_lbl_dir=stage / "labels" / "train",
        enforce_overlap=not args.no_non_overlap,
        min_area_640=args.min_area_640,
        dup_hamming_max=args.dup_hamming_max,
        forbidden_hashes=val_stats["hashes"],
        leak_hamming_max=args.leak_hamming_max,
        max_width=args.max_width,
    )

    # 3. Write canonical data.yaml & zip
    write_data_yaml(stage / "data.yaml", train_rel="images/train", val_rel="images/val")
    zip_path = package_dataset_zip(stage, args.out_zip)

    # 4. Generate QC report + visual preview contact sheets
    report_dir = args.out_zip.parent / f"{args.out_zip.stem}_qc_report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_text = build_qc_report(train_stats, val_stats, report_dir / "qc_report.md", zip_path)

    rng = random.Random(args.seed)
    t_sample = rng.sample(train_stats["saved_pairs"], min(16, len(train_stats["saved_pairs"])))
    v_sample = rng.sample(val_stats["saved_pairs"], min(16, len(val_stats["saved_pairs"])))
    render_contact_sheet(t_sample, report_dir / "qc_preview_train.jpg")
    render_contact_sheet(v_sample, report_dir / "qc_preview_val.jpg")

    print("\n" + report_text)
    print(f"[done] Packaged clean dataset -> {zip_path}")
    print(f"[done] QC report & previews  -> {report_dir}")


if __name__ == "__main__":
    main()
