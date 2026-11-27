#!/usr/bin/env python3
"""Script #1: Vietnam Street Frame Harvester (100% indoor, 0% Train/Val leakage).

Harvests sharp, dHash-deduplicated Vietnamese street images from:
  1. YouTube 4K dashcam POV URLs or local .mp4 videos (`configs/vn_youtube_sources.yaml`)
     with strict scene isolation between `val_videos` and `train_videos`.
  2. HuggingFace `star092304/Traffic-sign-detection-VietNam` (downloads `traffic-sign-detection-vietnam.zip`
     directly via `huggingface_hub.hf_hub_download` — public, non-gated, CC-BY-NC-4.0)
     and extracts both raw Vietnamese street images AND their exact GT traffic-sign boxes
     (remapped to competition class ID `9` = Sign) for SAM2 box-prompting in Script #2!

Usage:
    python scripts/01_vn_frame_harvester.py --config configs/vn_youtube_sources.yaml --out data/raw/vn_frames
    python scripts/01_vn_frame_harvester.py --hf-vn-signs --max-hf-images 350 --out data/raw/vn_frames
"""
from __future__ import annotations

import argparse
import csv
import io
import random
import shutil
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.seg_utils import IMG_EXTS, compute_dhash, hamming_distance  # noqa: E402


def compute_sharpness(bgr_img: np.ndarray) -> float:
    import cv2

    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def resize_max_width_pil(img: Image.Image, max_width: int) -> Image.Image:
    if max_width > 0 and img.width > max_width:
        new_h = max(1, int(round(img.height * (max_width / float(img.width)))))
        return img.resize((max_width, new_h), Image.Resampling.LANCZOS)
    return img


def download_youtube_video(url: str, cache_dir: Path, name: str) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    for p in cache_dir.glob(f"{name}.*"):
        if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"} and p.stat().st_size > 100_000:
            print(f"  [cache] Reusing {p.name}")
            return p

    if shutil.which("yt-dlp") is None:
        raise SystemExit("yt-dlp not found. Install with: pip install yt-dlp")

    out_tpl = str(cache_dir / f"{name}.%(ext)s")
    cmd = [
        "yt-dlp",
        "-f", "bestvideo[height<=1080][ext=mp4]/best[height<=1080][ext=mp4]/best[height<=1080]",
        "--no-playlist",
        "--retries", "3",
        "-o", out_tpl,
        url,
    ]
    print(f"  [yt-dlp] Downloading {name} ({url}) ...")
    subprocess.run(cmd, check=True)
    matches = [p for p in cache_dir.glob(f"{name}.*") if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"}]
    if not matches:
        raise FileNotFoundError(f"yt-dlp finished but no video found for {name}")
    return matches[0]


def harvest_video(
    video_path: Path,
    source_label: str,
    source_url: str,
    split: str,
    out_dir: Path,
    known_hashes: list[int],
    interval_sec: float = 4.5,
    min_sharpness: float = 55.0,
    min_dhash_dist: int = 8,
    max_width: int = 1280,
    jpeg_quality: int = 92,
    skip_start_sec: float = 15.0,
    skip_end_sec: float = 15.0,
    max_frames: int = 200,
    scenario: str = "",
) -> list[dict[str, Any]]:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    duration_sec = total_frames / fps if fps > 0 else 0.0
    end_sec = max(skip_start_sec, duration_sec - skip_end_sec) if duration_sec > 0 else 1e9

    img_out = out_dir / split / "images"
    img_out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    saved = 0
    skipped_blur = 0
    skipped_dup = 0
    cur_sec = skip_start_sec

    while cur_sec <= end_sec and saved < max_frames:
        cap.set(cv2.CAP_PROP_POS_MSEC, cur_sec * 1000.0)
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        sharp = compute_sharpness(frame)
        if sharp < min_sharpness:
            skipped_blur += 1
            cur_sec += interval_sec
            continue

        pil_rgb = resize_max_width_pil(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)), max_width)
        h_val = compute_dhash(pil_rgb)
        if any(hamming_distance(h_val, kh) <= min_dhash_dist for kh in known_hashes):
            skipped_dup += 1
            cur_sec += interval_sec
            continue

        fname = f"{source_label}_{int(cur_sec):05d}s_{saved:04d}.jpg"
        pil_rgb.save(img_out / fname, quality=jpeg_quality)
        known_hashes.append(h_val)
        rows.append({
            "file": fname,
            "split": split,
            "source_name": source_label,
            "source_url": source_url,
            "scenario": scenario,
            "timestamp_sec": round(cur_sec, 2),
            "sharpness": round(sharp, 1),
            "dhash_hex": f"{h_val:016x}",
        })
        saved += 1
        cur_sec += interval_sec

    cap.release()
    print(f"  [{split}] {source_label}: saved={saved}, blur_skip={skipped_blur}, dup_skip={skipped_dup}")
    return rows


def harvest_hf_vn_signs(
    max_images: int,
    split: str,
    out_dir: Path,
    known_hashes: list[int],
    min_dhash_dist: int = 8,
    max_width: int = 1280,
    jpeg_quality: int = 92,
    seed: int = 23,
    hf_repo: str = "star092304/Traffic-sign-detection-VietNam",
    zip_filename: str = "traffic-sign-detection-vietnam.zip",
) -> list[dict[str, Any]]:
    """Download `traffic-sign-detection-vietnam.zip` from HuggingFace (public, 905MB) and extract VN images + Sign boxes."""
    from huggingface_hub import hf_hub_download

    print(f"[HF] Downloading {hf_repo}/{zip_filename} (public, CC-BY-NC-4.0)...")
    zip_local = Path(hf_hub_download(repo_id=hf_repo, filename=zip_filename, repo_type="dataset"))
    print(f"[HF] Downloaded zip: {zip_local} ({zip_local.stat().st_size / 1e6:.1f} MB)")

    img_out = out_dir / split / "images"
    box_out = out_dir / split / "extra_sign_boxes"
    img_out.mkdir(parents=True, exist_ok=True)
    box_out.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    saved = 0
    with zipfile.ZipFile(zip_local) as zf:
        all_names = zf.namelist()
        # Index .txt label entries by stem
        lbl_map: dict[str, str] = {}
        img_entries: list[str] = []
        for name in all_names:
            low = name.lower()
            if name.endswith("/") or "__macosx" in low:
                continue
            p = Path(name)
            if p.suffix.lower() in IMG_EXTS:
                # Keep train/valid splits distinct if harvesting val vs train
                if split == "val" and "/valid/" in low:
                    img_entries.append(name)
                elif split == "train" and "/train/" in low:
                    img_entries.append(name)
                elif "/train/" not in low and "/valid/" not in low:
                    img_entries.append(name)
            elif p.suffix.lower() == ".txt" and p.name.lower() not in {"classes.txt", "readme.txt"}:
                lbl_map[p.stem] = name

        if not img_entries:
            img_entries = [n for n in all_names if not n.endswith("/") and Path(n).suffix.lower() in IMG_EXTS]

        rng = random.Random(seed)
        rng.shuffle(img_entries)
        print(f"[HF] Found {len(img_entries)} candidate VN street images in zip (with {len(lbl_map)} GT sign label files)")

        for idx, entry in enumerate(img_entries):
            if saved >= max_images:
                break
            try:
                raw_bytes = zf.read(entry)
                with Image.open(io.BytesIO(raw_bytes)) as im:
                    rgb = resize_max_width_pil(ImageOps.exif_transpose(im).convert("RGB"), max_width)
            except Exception:  # noqa: BLE001
                continue

            # Skip tiny cropped sign icons — keep only full road scenes (>= 480 px wide)
            if rgb.width < 480 or rgb.height < 300:
                continue

            h_val = compute_dhash(rgb)
            if any(hamming_distance(h_val, kh) <= min_dhash_dist for kh in known_hashes):
                continue

            stem_orig = Path(entry).stem
            fname = f"hf_vnsign_{saved:05d}.jpg"
            rgb.save(img_out / fname, quality=jpeg_quality)
            known_hashes.append(h_val)

            # Remap ALL 82 Vietnamese road-sign classes -> competition class 9 (Sign)
            lbl_entry = lbl_map.get(stem_orig)
            if lbl_entry:
                sign_lines: list[str] = []
                for line in zf.read(lbl_entry).decode("utf-8", errors="replace").splitlines():
                    parts = line.strip().split()
                    if len(parts) == 5:
                        cx, cy, bw, bh = [float(v) for v in parts[1:]]
                        if bw > 0.006 and bh > 0.006:
                            sign_lines.append(f"9 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
                if sign_lines:
                    (box_out / f"{Path(fname).stem}.txt").write_text("\n".join(sign_lines) + "\n", encoding="utf-8")

            rows.append({
                "file": fname,
                "split": split,
                "source_name": "hf_vn_traffic_sign",
                "source_url": f"https://huggingface.co/datasets/{hf_repo}",
                "scenario": "vn_street_with_gt_signs",
                "timestamp_sec": -1,
                "sharpness": -1,
                "dhash_hex": f"{h_val:016x}",
            })
            saved += 1

    print(f"  [{split}] HF VN Signs: saved={saved} full VN street images (+ GT sign boxes in {box_out})")
    return rows


def append_manifest(manifest_path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["file", "split", "source_name", "source_url", "scenario", "timestamp_sec", "sharpness", "dhash_hex"]
    write_header = not manifest_path.exists()
    with open(manifest_path, "a", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            w.writeheader()
        w.writerows(rows)


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    ap = argparse.ArgumentParser(description="Harvest clean, unique Vietnamese street frames with 0% Val/Train leakage.")
    ap.add_argument("--config", type=Path, default=None, help="Path to vn_youtube_sources.yaml")
    ap.add_argument("--video", type=str, default=None, help="Single local video path or YouTube URL")
    ap.add_argument("--split", choices=["val", "train"], default="train")
    ap.add_argument("--name", type=str, default="vn_clip")
    ap.add_argument("--max-frames", type=int, default=200)
    ap.add_argument("--hf-vn-signs", action="store_true", help="Also harvest real VN street images + GT sign boxes from HF")
    ap.add_argument("--max-hf-images", type=int, default=300)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, default=None)
    args = ap.parse_args()

    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    known_hashes: list[int] = []
    manifest_path = out_dir / "harvest_manifest.csv"
    if manifest_path.is_file():
        with open(manifest_path, "r", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("dhash_hex"):
                    known_hashes.append(int(row["dhash_hex"], 16))
        print(f"Loaded {len(known_hashes)} existing dHashes from {manifest_path}")

    all_rows: list[dict[str, Any]] = []
    cache_dir = args.cache_dir or Path(tempfile.mkdtemp(prefix="vn_yt_cache_"))

    if args.config is not None:
        import yaml

        cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
        st = cfg.get("settings", {})
        for split_key, split_name in [("val_videos", "val"), ("train_videos", "train")]:
            for item in (cfg.get(split_key) or []):
                src = str(item["source"])
                name = str(item["name"])
                try:
                    vpath = download_youtube_video(src, cache_dir, name) if src.startswith("http") else Path(src)
                    rows = harvest_video(
                        video_path=vpath,
                        source_label=name,
                        source_url=src,
                        split=split_name,
                        out_dir=out_dir,
                        known_hashes=known_hashes,
                        interval_sec=float(st.get("interval_sec", 4.5)),
                        min_sharpness=float(st.get("min_sharpness", 55.0)),
                        min_dhash_dist=int(st.get("min_dhash_dist", 8)),
                        max_width=int(st.get("max_width", 1280)),
                        jpeg_quality=int(st.get("jpeg_quality", 92)),
                        skip_start_sec=float(st.get("skip_start_sec", 15.0)),
                        skip_end_sec=float(st.get("skip_end_sec", 15.0)),
                        max_frames=int(item.get("max_frames", args.max_frames)),
                        scenario=str(item.get("scenario", "")),
                    )
                    all_rows.extend(rows)
                except Exception as exc:  # noqa: BLE001
                    print(f"  [warn] Skipped {name} ({src}): {exc}")

    if args.video is not None:
        vpath = download_youtube_video(args.video, cache_dir, args.name) if args.video.startswith("http") else Path(args.video)
        all_rows.extend(
            harvest_video(
                video_path=vpath,
                source_label=args.name,
                source_url=args.video,
                split=args.split,
                out_dir=out_dir,
                known_hashes=known_hashes,
                max_frames=args.max_frames,
            )
        )

    if args.hf_vn_signs:
        all_rows.extend(
            harvest_hf_vn_signs(
                max_images=args.max_hf_images,
                split=args.split,
                out_dir=out_dir,
                known_hashes=known_hashes,
            )
        )

    append_manifest(manifest_path, all_rows)
    print(f"\nDONE! Total newly harvested frames: {len(all_rows)} -> {out_dir}")


if __name__ == "__main__":
    main()
