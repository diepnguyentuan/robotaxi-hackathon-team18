"""Unit & integration tests verifying core Data Engine logic (Rider-Bike Union, <64px², dHash, Auto-QC packager)."""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from common.seg_utils import (
    CLASSES,
    MIN_GT_AREA_PX,
    clean_and_normalize_polygon,
    compute_dhash,
    enforce_non_overlap,
    hamming_distance,
    merge_riders_and_vehicles,
    poly_area_at_640,
    read_yolo_seg_txt,
    write_yolo_seg_txt,
)


def rect_poly(x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    return np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)


class TestSegUtilsAndQC(unittest.TestCase):
    def test_area_at_640_exact_match_with_guideline(self) -> None:
        # On 1280x720 image, r = 640/1280 = 0.5 -> 16x16 px box = 256 px² original = 64 px² at 640 scale!
        w, h = 1280, 720
        box_16px = rect_poly(0.1, 0.1, 0.1 + 16.0 / w, 0.1 + 16.0 / h)
        self.assertAlmostEqual(poly_area_at_640(box_16px, w, h), 64.0, places=3)

        # Tiny 10x10 px box on 1280x720 -> 25 px² at 640 (<64 -> must be dropped!)
        tiny = rect_poly(0.1, 0.1, 0.1 + 10.0 / w, 0.1 + 10.0 / h)
        self.assertIsNone(clean_and_normalize_polygon(tiny, w, h, min_area_640=MIN_GT_AREA_PX))

    def test_dhash_identical_vs_different(self) -> None:
        img1 = Image.new("RGB", (320, 180), (40, 40, 40))
        ImageDraw.Draw(img1).rectangle([20, 20, 160, 150], fill=(220, 220, 220))
        img2 = img1.copy()
        img3 = Image.new("RGB", (320, 180), (220, 220, 220))
        ImageDraw.Draw(img3).rectangle([160, 20, 300, 150], fill=(10, 10, 10))

        self.assertEqual(hamming_distance(compute_dhash(img1), compute_dhash(img2)), 0)
        self.assertGreater(hamming_distance(compute_dhash(img1), compute_dhash(img3)), 8)

    def test_explicit_rider_multi_person_union(self) -> None:
        # 1 motorcycle + 2 riders (driver + passenger) + 1 parked motorcycle -> 1 Motorcyclist (5) + 1 Motorcycle (4)
        w, h = 1280, 720
        raw = [
            {"role": 4, "poly": rect_poly(0.30, 0.50, 0.45, 0.72), "score": 1.0},             # ridden motorbike
            {"role": "rider", "poly": rect_poly(0.33, 0.34, 0.40, 0.62), "score": 1.0},       # driver
            {"role": "rider", "poly": rect_poly(0.38, 0.35, 0.44, 0.60), "score": 1.0},       # passenger
            {"role": 4, "poly": rect_poly(0.75, 0.55, 0.85, 0.72), "score": 1.0},             # parked motorbike
            {"role": 6, "poly": rect_poly(0.10, 0.55, 0.20, 0.72), "score": 1.0},             # bicycle
            {"role": "bicyclist_rider", "poly": rect_poly(0.12, 0.38, 0.18, 0.64), "score": 1.0},
        ]
        merged = merge_riders_and_vehicles(raw, mode="explicit_rider", w=w, h=h)
        cls_ids = sorted(it["class_id"] for it in merged)
        self.assertEqual(cls_ids, [4, 5, 7])

    def test_generic_person_distinguishes_rider_from_walking_pedestrian(self) -> None:
        w, h = 1280, 720
        raw = [
            # Motorbike + seated person on top -> Motorcyclist (5)
            {"role": 4, "poly": rect_poly(0.30, 0.50, 0.44, 0.70), "score": 0.9},
            {"role": 8, "poly": rect_poly(0.33, 0.35, 0.41, 0.65), "score": 0.95},
            # Parked motorbike + pedestrian walking FAR to the right -> Motorcycle (4) + Pedestrian (8)
            {"role": 4, "poly": rect_poly(0.65, 0.52, 0.75, 0.70), "score": 0.9},
            {"role": 8, "poly": rect_poly(0.82, 0.40, 0.88, 0.70), "score": 0.95},
        ]
        merged = merge_riders_and_vehicles(raw, mode="generic_person", w=w, h=h)
        cls_ids = sorted(it["class_id"] for it in merged)
        self.assertEqual(cls_ids, [4, 5, 8])

    def test_same_class_dedup_in_non_overlap(self) -> None:
        w, h = 1280, 720
        items = [
            {"class_id": 0, "poly": rect_poly(0.20, 0.40, 0.50, 0.70), "score": 0.95},
            {"class_id": 0, "poly": rect_poly(0.21, 0.41, 0.49, 0.69), "score": 0.70},  # duplicate Car
        ]
        out = enforce_non_overlap(items, w=w, h=h)
        self.assertEqual(len(out), 1)

    def test_end_to_end_qc_packager_blocks_val_leakage_and_tiny_masks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            val_dir = root / "val_src"
            trn_dir = root / "trn_src"
            (val_dir / "images").mkdir(parents=True)
            (val_dir / "labels").mkdir(parents=True)
            (trn_dir / "images").mkdir(parents=True)
            (trn_dir / "labels").mkdir(parents=True)

            # Create a distinct Golden Val image
            val_img = Image.new("RGB", (640, 360), (30, 60, 90))
            ImageDraw.Draw(val_img).rectangle([50, 50, 300, 260], fill=(240, 200, 40))
            val_img.save(val_dir / "images" / "v01.jpg", quality=95)
            write_yolo_seg_txt(val_dir / "labels" / "v01.txt", [
                {"class_id": 5, "poly": rect_poly(0.2, 0.2, 0.5, 0.6)},
            ])

            # Train image 1: EXACT LEAK of val_img -> MUST BE BLOCKED!
            val_img.save(trn_dir / "images" / "t_leak.jpg", quality=95)
            write_yolo_seg_txt(trn_dir / "labels" / "t_leak.txt", [
                {"class_id": 5, "poly": rect_poly(0.2, 0.2, 0.5, 0.6)},
            ])

            # Train image 2: Distinct valid image with 1 valid polygon + 1 tiny <64px² polygon
            trn_img = Image.new("RGB", (640, 360), (200, 200, 200))
            ImageDraw.Draw(trn_img).ellipse([320, 40, 590, 310], fill=(20, 120, 40))
            trn_img.save(trn_dir / "images" / "t_ok.jpg", quality=95)
            write_yolo_seg_txt(trn_dir / "labels" / "t_ok.txt", [
                {"class_id": 3, "poly": rect_poly(0.3, 0.3, 0.7, 0.7)},       # valid Other vehicle
                {"class_id": 9, "poly": rect_poly(0.01, 0.01, 0.015, 0.015)}, # tiny -> dropped!
            ])

            out_zip = root / "dataset_test.zip"
            cmd = [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "04_auto_qc_and_packager.py"),
                "--train-dirs", str(trn_dir),
                "--val-dir", str(val_dir),
                "--out-zip", str(out_zip),
            ]
            res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            self.assertEqual(res.returncode, 0, f"STDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}")
            self.assertTrue(out_zip.is_file(), res.stderr)

            with zipfile.ZipFile(out_zip) as zf:
                names = set(zf.namelist())
                self.assertIn("data.yaml", names)
                train_imgs = [n for n in names if n.startswith("images/train/")]
                val_imgs = [n for n in names if n.startswith("images/val/")]
                self.assertEqual(len(val_imgs), 1)
                self.assertEqual(len(train_imgs), 1, f"Expected leaked train image to be dropped; got {train_imgs}")
                # Check tiny polygon was stripped from surviving train label
                lbl_name = [n for n in names if n.startswith("labels/train/")][0]
                lbl_lines = [l for l in zf.read(lbl_name).decode("utf-8").splitlines() if l.strip()]
                self.assertEqual(len(lbl_lines), 1)
                self.assertTrue(lbl_lines[0].startswith("3 "))


if __name__ == "__main__":
    unittest.main(verbosity=2)
