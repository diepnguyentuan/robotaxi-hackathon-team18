#!/usr/bin/env python3
"""Script #2: High-Precision Pre-Label Factory (YOLO-seg + YOLO-World + SAM2 -> CVAT).

Runs on Kaggle/Colab GPU to auto-generate tight 11-class polygon pre-labels for Vietnamese
street images, including:
  1. Base instance segmentation (yolo11x-seg.pt) for vehicles, pedestrians, traffic lights
  2. Optional Open-Vocab booster (yolov8x-worldv2.pt) for Vietnam-specific Signs & Other vehicle
     (ba gác/rickshaw, cyclo, trailer/container, construction vehicles)
  3. Optional external Sign bboxes (--extra-sign-boxes, e.g. from HF VN Sign dataset)
  4. Optional SAM2 (sam2.1_b.pt) box-prompt mask refinement for IoU 0.75-0.95 boundary precision
  5. Geometric Rider-Vehicle Union (Motorcyclist=5, Cyclist=7) + Z-order Non-Overlap + <64px² filter
  6. Direct CVAT-compatible YOLO Segmentation 1.0 export zip!

Usage (on Kaggle / Colab GPU):
    python scripts/02_auto_prelabel_sam2.py \
        --images data/raw/vn_frames/val \
        --out data/processed/cvat_val_prelabel \
        --use-yoloworld --use-sam2
"""
from __future__ import annotations

import argparse
import shutil
import sys
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common.seg_utils import (  # noqa: E402
    CLASSES,
    IMG_EXTS,
    MIN_GT_AREA_PX,
    Instance,
    clean_polygon,
    enforce_non_overlap,
    merge_riders_and_vehicles,
    render_contact_sheet,
    write_data_yaml,
    write_yolo_seg_file,
)

# COCO class id -> Competition class id
COCO_TO_COMP: dict[int, int] = {
    0: 8,   # person -> Pedestrian (later merged with bike -> Motorcyclist/Cyclist if seated)
    1: 6,   # bicycle -> Bicycle
    2: 0,   # car -> Car
    3: 4,   # motorcycle -> Motorcycle
    5: 1,   # bus -> Bus
    7: 2,   # truck -> Truck
    9: 10,  # traffic light -> Traffic light
    11: 9,  # stop sign -> Sign
}

# Open-vocabulary prompts for classes underrepresented in COCO
WORLD_PROMPTS: list[tuple[str, int]] = [
    ("road traffic sign", 9),
    ("street sign plate", 9),
    ("auto rickshaw", 3),
    ("tuk tuk", 3),
    ("cyclo tricycle", 3),
    ("three-wheeled cargo cart", 3),
    ("excavator bulldozer", 3),
    ("forklift crane", 3),
    ("semi-trailer shipping container", 3),
    ("traffic light box", 10),
]


def mask_to_largest_polygon(mask_bin: np.ndarray, width: int, height: int) -> np.ndarray | None:
    """Convert binary HxW mask to normalized (N, 2) exterior contour via OpenCV."""
    import cv2

    u8 = (mask_bin > 0).astype(np.uint8)
    contours, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    largest = max(contours, key=cv2.contourArea)
    if largest.shape[0] < 3:
        return None
    pts = largest.reshape(-1, 2).astype(np.float64)
    pts[:, 0] /= float(width)
    pts[:, 1] /= float(height)
    return pts


def load_extra_sign_boxes(box_dir: Path, stem: str, width: int, height: int) -> list[tuple[int, np.ndarray, float]]:
    """Read normalized YOLO bbox .txt (cls cx cy w h) -> (class_id=9, xyxy_px, score=0.95)."""
    p = box_dir / f"{stem}.txt"
    if not p.is_file():
        return []
    out = []
    for line in p.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        cx, cy, bw, bh = [float(v) for v in parts[1:5]]
        x1 = max(0.0, (cx - bw / 2.0) * width)
        y1 = max(0.0, (cy - bh / 2.0) * height)
        x2 = min(float(width), (cx + bw / 2.0) * width)
        y2 = min(float(height), (cy + bh / 2.0) * height)
        if x2 - x1 > 4 and y2 - y1 > 4:
            out.append((9, np.array([x1, y1, x2, y2], dtype=np.float32), 0.95))
    return out


def package_cvat_zip(out_dir: Path, zip_path: Path) -> Path:
    """Package images + labels + data.yaml + obj.names for CVAT YOLO Segmentation 1.0 import."""
    (out_dir / "obj.names").write_text("\n".join(CLASSES) + "\n", encoding="utf-8")
    write_data_yaml(out_dir / "data.yaml", train_rel="images/train", val_rel="images/train")
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for sub in ("data.yaml", "obj.names"):
            zf.write(out_dir / sub, arcname=sub)
        for folder in ("images/train", "labels/train"):
            d = out_dir / folder
            if d.is_dir():
                for fp in sorted(d.iterdir()):
                    if fp.is_file():
                        zf.write(fp, arcname=f"{folder}/{fp.name}")
    return zip_path


def main() -> None:
    ap = argparse.ArgumentParser(description="Auto Pre-Label VN images with YOLO-seg + YOLO-World + SAM2 -> CVAT.")
    ap.add_argument("--images", type=Path, required=True, help="Directory containing input images")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for CVAT import package")
    ap.add_argument("--yolo-model", type=str, default="yolo11x-seg.pt", help="Base Ultralytics seg model")
    ap.add_argument("--use-yoloworld", action="store_true", help="Boost Sign & Other vehicle with YOLO-Worldv2")
    ap.add_argument("--world-model", type=str, default="yolov8x-worldv2.pt")
    ap.add_argument("--use-sam2", action="store_true", help="Refine all candidate boxes with SAM2.1-b masks")
    ap.add_argument("--sam-model", type=str, default="sam2.1_b.pt")
    ap.add_argument("--extra-sign-boxes", type=Path, default=None, help="Optional dir of YOLO bbox .txt for Signs")
    ap.add_argument("--conf", type=float, default=0.25, help="Base YOLO confidence threshold")
    ap.add_argument("--world-conf", type=float, default=0.20, help="YOLO-World confidence threshold")
    ap.add_argument("--device", type=str, default="0", help="GPU device id or 'cpu'")
    ap.add_argument("--max-images", type=int, default=0, help="0 = process all images")
    args = ap.parse_args()

    from ultralytics import SAM, YOLO, YOLOWorld

    image_paths = sorted(p for p in args.images.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS)
    if args.max_images > 0:
        image_paths = image_paths[: args.max_images]
    if not image_paths:
        raise SystemExit(f"No images found in {args.images}")

    print(f"[init] Loading base segmenter: {args.yolo_model}")
    base_model = YOLO(args.yolo_model)

    world_model = None
    prompt_to_cls: dict[int, int] = {}
    if args.use_yoloworld:
        print(f"[init] Loading open-vocab booster: {args.world_model}")
        world_model = YOLOWorld(args.world_model)
        prompt_texts = [p for p, _ in WORLD_PROMPTS]
        prompt_to_cls = {i: c for i, (_, c) in enumerate(WORLD_PROMPTS)}
        world_model.set_classes(prompt_texts)

    sam_model = None
    if args.use_sam2:
        print(f"[init] Loading SAM2 refiner: {args.sam_model}")
        sam_model = SAM(args.sam_model)

    img_out = args.out / "images" / "train"
    lbl_out = args.out / "labels" / "train"
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    preview_pairs: list[tuple[Path, list[Instance]]] = []
    total_instances = 0

    for idx, img_path in enumerate(image_paths, start=1):
        with Image.open(img_path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            w, h = im.size
            dst_img = img_out / f"{img_path.stem}.jpg"
            im.save(dst_img, quality=92)

        raw_instances: list[Instance] = []
        box_candidates: list[tuple[int, np.ndarray, float]] = []

        # 1. Base YOLO-seg pass
        res = base_model.predict(str(dst_img), conf=args.conf, imgsz=1024, device=args.device, verbose=False)[0]
        if res.boxes is not None and len(res.boxes) > 0:
            cls_arr = res.boxes.cls.cpu().numpy().astype(int)
            conf_arr = res.boxes.conf.cpu().numpy()
            xyxy_arr = res.boxes.xyxy.cpu().numpy()
            masks_xyn = res.masks.xyn if res.masks is not None else [None] * len(cls_arr)
            for c_coco, sc, xyxy, poly in zip(cls_arr, conf_arr, xyxy_arr, masks_xyn):
                if c_coco not in COCO_TO_COMP:
                    continue
                comp_cls = COCO_TO_COMP[c_coco]
                if sam_model is not None:
                    box_candidates.append((comp_cls, xyxy, float(sc)))
                elif poly is not None and len(poly) >= 3:
                    cleaned = clean_polygon(poly, w, h, min_area_640=MIN_GT_AREA_PX)
                    if cleaned is not None:
                        raw_instances.append(Instance(class_id=comp_cls, polygon=cleaned, score=float(sc)))

        # 2. Optional YOLO-World pass for Sign & Other vehicle
        if world_model is not None:
            wres = world_model.predict(str(dst_img), conf=args.world_conf, imgsz=1024, device=args.device, verbose=False)[0]
            if wres.boxes is not None and len(wres.boxes) > 0:
                wcls = wres.boxes.cls.cpu().numpy().astype(int)
                wconf = wres.boxes.conf.cpu().numpy()
                wxyxy = wres.boxes.xyxy.cpu().numpy()
                for wi, sc, xyxy in zip(wcls, wconf, wxyxy):
                    comp_cls = prompt_to_cls.get(int(wi))
                    if comp_cls is None:
                        continue
                    box_candidates.append((comp_cls, xyxy, float(sc)))

        # 3. Optional external GT Sign boxes (e.g. HF VN Traffic Sign dataset)
        if args.extra_sign_boxes is not None:
            box_candidates.extend(load_extra_sign_boxes(args.extra_sign_boxes, img_path.stem, w, h))

        # 4. SAM2 box-prompt mask generation
        if box_candidates:
            if sam_model is not None:
                bboxes_np = np.stack([b[1] for b in box_candidates], axis=0)
                sres = sam_model.predict(str(dst_img), bboxes=bboxes_np, device=args.device, verbose=False)[0]
                if sres.masks is not None:
                    masks_data = sres.masks.data.cpu().numpy()
                    for (comp_cls, _, sc), m_bin in zip(box_candidates, masks_data):
                        poly = mask_to_largest_polygon(m_bin, m_bin.shape[1], m_bin.shape[0])
                        if poly is not None:
                            cleaned = clean_polygon(poly, w, h, min_area_640=MIN_GT_AREA_PX)
                            if cleaned is not None:
                                raw_instances.append(Instance(class_id=comp_cls, polygon=cleaned, score=sc))
            else:
                # Without SAM2, approximate world/extra boxes as tight rectangles (human refines in CVAT)
                for comp_cls, xyxy, sc in box_candidates:
                    x1, y1, x2, y2 = xyxy[0] / w, xyxy[1] / h, xyxy[2] / w, xyxy[3] / h
                    rect = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float64)
                    cleaned = clean_polygon(rect, w, h, min_area_640=MIN_GT_AREA_PX)
                    if cleaned is not None:
                        raw_instances.append(Instance(class_id=comp_cls, polygon=cleaned, score=sc))

        # 5. Rider-Vehicle Union (Motorcyclist=5, Cyclist=7) + Non-Overlap QC
        merged = merge_riders_and_vehicles(raw_instances, w, h, mode="generic_person", min_area_640=MIN_GT_AREA_PX)
        final_insts = enforce_non_overlap(merged, w, h, min_area_640=MIN_GT_AREA_PX)

        write_yolo_seg_file(lbl_out / f"{img_path.stem}.txt", final_insts)
        total_instances += len(final_insts)
        if len(preview_pairs) < 16 and final_insts:
            preview_pairs.append((dst_img, final_insts))

        if idx % 25 == 0 or idx == len(image_paths):
            print(f"[prelabel] {idx}/{len(image_paths)} images done (total instances: {total_instances})")

    # Visual QC preview + CVAT import zip
    render_contact_sheet(preview_pairs, args.out / "prelabel_preview.jpg")
    zip_path = package_cvat_zip(args.out, args.out / "cvat_import_yolo_seg.zip")
    print(f"[done] Pre-label CVAT zip ready: {zip_path} ({len(image_paths)} images, {total_instances} instances)")


if __name__ == "__main__":
    main()
