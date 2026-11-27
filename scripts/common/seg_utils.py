"""Core geometry, Rider-Vehicle Union, Z-Order Non-Overlap, dHash, and YOLO-seg I/O utilities.

Strictly aligned with `docs/official/train_and_export.ipynb` (detscore v0.2.0):
  - Fixed 11 competition classes (IDs 0..10)
  - Letterbox imgsz = 640, MIN_GT_AREA_PX = 64.0
  - 64-bit perceptual dHash compatible with Step 2 leakage audit
  - Shortest-bridge MultiPolygon merger (keeps split-occluded objects as ONE single instance)
  - Rider-Vehicle Union (supports multi-rider motorbikes & both explicit-rider and generic-person modes)
  - Z-Order Non-Overlap enforcement (foreground keeps pixels, occluded behind object gets visible remainder)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

CLASSES: tuple[str, ...] = (
    "Car",
    "Bus",
    "Truck",
    "Other vehicle",
    "Motorcycle",
    "Motorcyclist",
    "Bicycle",
    "Cyclist",
    "Pedestrian",
    "Sign",
    "Traffic light",
)

CLASS_DISPLAY_VI: dict[str, str] = {
    "Car": "Ô tô con",
    "Bus": "Xe buýt / xe khách",
    "Truck": "Xe tải",
    "Other vehicle": "Phương tiện khác",
    "Motorcycle": "Xe máy (không người)",
    "Motorcyclist": "Xe máy (có người)",
    "Bicycle": "Xe đạp (không người)",
    "Cyclist": "Xe đạp (có người)",
    "Pedestrian": "Người đi bộ",
    "Sign": "Biển báo",
    "Traffic light": "Đèn giao thông",
}

IMGSZ: int = 640
MIN_GT_AREA_PX: float = 64.0
IMG_EXTS: set[str] = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CLASS_COLORS: tuple[tuple[int, int, int], ...] = (
    (52, 152, 219),   # 0 Car
    (155, 89, 182),   # 1 Bus
    (230, 126, 34),   # 2 Truck
    (241, 196, 15),   # 3 Other vehicle
    (26, 188, 156),   # 4 Motorcycle
    (46, 204, 113),   # 5 Motorcyclist
    (52, 73, 94),     # 6 Bicycle
    (22, 160, 133),   # 7 Cyclist
    (231, 76, 60),    # 8 Pedestrian
    (243, 156, 18),   # 9 Sign
    (233, 30, 99),    # 10 Traffic light
)

try:
    from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
    from shapely.ops import unary_union
    from shapely.validation import make_valid

    HAS_SHAPELY = True
except ImportError:
    HAS_SHAPELY = False


@dataclass
class Instance:
    """Single segmentation instance with normalized (N, 2) polygon in [0, 1]."""

    class_id: int
    polygon: np.ndarray
    role: str | None = None
    score: float = 1.0
    source_tag: str = ""


# --------------------------------------------------------------------------- Perceptual dHash
def compute_dhash(img: Image.Image) -> int:
    """Compute 64-bit horizontal difference hash (9x8 grayscale -> 64 bits)."""
    gray = img.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
    arr = np.asarray(gray, dtype=np.int16)
    diff = arr[:, 1:] > arr[:, :-1]
    bits = 0
    for b in diff.flatten():
        bits = (bits << 1) | int(bool(b))
    return bits


def hamming_distance(a: int, b: int) -> int:
    """Bit-level Hamming distance between two 64-bit dHashes."""
    return int((a ^ b).bit_count())


# --------------------------------------------------------------------------- Polygon Math
def poly_area_norm(pts: np.ndarray) -> float:
    """Shoelace area of a normalized (N, 2) polygon in [0, 1]^2."""
    if pts is None or len(pts) < 3:
        return 0.0
    x = pts[:, 0].astype(np.float64)
    y = pts[:, 1].astype(np.float64)
    return 0.5 * float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def poly_area_at_640(pts: np.ndarray, w: int, h: int, imgsz: int = IMGSZ) -> float:
    """Exact mask area in square pixels at competition letterboxed 640x640 scale."""
    if w <= 0 or h <= 0:
        return 0.0
    r = min(float(imgsz) / float(w), float(imgsz) / float(h))
    return poly_area_norm(pts) * float(w) * float(h) * (r * r)


def poly_bbox(pts: np.ndarray) -> tuple[float, float, float, float]:
    """Axis-aligned normalized bounding box (x1, y1, x2, y2)."""
    xs = pts[:, 0]
    ys = pts[:, 1]
    return float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())


def bbox_iou(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> float:
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    union = a1 + a2 - inter
    return inter / union if union > 0.0 else 0.0


def bbox_ios(b1: tuple[float, float, float, float], b2: tuple[float, float, float, float]) -> float:
    """Intersection over smaller box area."""
    ix1, iy1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    ix2, iy2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    a1 = max(0.0, (b1[2] - b1[0]) * (b1[3] - b1[1]))
    a2 = max(0.0, (b2[2] - b2[0]) * (b2[3] - b2[1]))
    denom = min(a1, a2)
    return inter / denom if denom > 0.0 else 0.0


def _bridge_two_rings(ring_a: np.ndarray, ring_b: np.ndarray) -> np.ndarray:
    """Connect two disjoint polygon rings via their shortest vertex bridge (zero-area cut-line)."""
    diff = ring_a[:, None, :] - ring_b[None, :, :]
    dist2 = np.sum(diff * diff, axis=-1)
    idx_a, idx_b = np.unravel_index(int(np.argmin(dist2)), dist2.shape)
    part1 = ring_a[: idx_a + 1]
    part2 = np.vstack([ring_b[idx_b:], ring_b[: idx_b + 1]])
    part3 = ring_a[idx_a:]
    return np.vstack([part1, part2, part3]).astype(np.float32)


def _extract_polygons(geom: Any) -> list[Any]:
    if geom is None or geom.is_empty:
        return []
    if isinstance(geom, Polygon):
        return [geom]
    if isinstance(geom, (MultiPolygon, GeometryCollection)):
        out: list[Any] = []
        for g in geom.geoms:
            out.extend(_extract_polygons(g))
        return out
    return []


def geom_to_single_ring(
    geom: Any,
    w: int,
    h: int,
    min_piece_area_640: float = 24.0,
    simplify_tol: float = 0.001,
) -> np.ndarray | None:
    """Convert a Shapely Polygon/MultiPolygon into ONE continuous YOLO-seg ring (N, 2)."""
    polys = _extract_polygons(geom)
    if not polys:
        return None
    polys.sort(key=lambda p: p.area, reverse=True)
    max_area = polys[0].area
    if max_area <= 0.0:
        return None

    r = min(float(IMGSZ) / float(w), float(IMGSZ) / float(h))
    scale_to_640 = float(w) * float(h) * (r * r)

    kept_rings: list[np.ndarray] = []
    for p in polys:
        if p.area * scale_to_640 < min_piece_area_640 and len(kept_rings) >= 1:
            continue
        if p.area < 0.05 * max_area and len(kept_rings) >= 1:
            continue
        sp = p.simplify(simplify_tol, preserve_topology=True) if simplify_tol > 0 else p
        if sp.is_empty or not isinstance(sp, Polygon):
            sp = p
        coords = np.asarray(sp.exterior.coords, dtype=np.float32)
        if len(coords) >= 4 and np.allclose(coords[0], coords[-1]):
            coords = coords[:-1]
        if len(coords) >= 3:
            kept_rings.append(np.clip(coords, 0.0, 1.0))

    if not kept_rings:
        return None
    ring = kept_rings[0]
    for nxt in kept_rings[1:]:
        ring = _bridge_two_rings(ring, nxt)
    return np.clip(ring, 0.0, 1.0)


def clean_and_normalize_polygon(
    pts: np.ndarray,
    w: int,
    h: int,
    min_area_640: float = MIN_GT_AREA_PX,
    simplify_tol: float = 0.001,
) -> np.ndarray | None:
    """Validate, clip to [0, 1], repair self-intersections, and enforce >=64 px²@640."""
    arr = np.asarray(pts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2 or len(arr) < 3:
        return None
    if not np.all(np.isfinite(arr)):
        return None
    arr = np.clip(arr, 0.0, 1.0)

    if HAS_SHAPELY:
        try:
            poly = Polygon(arr)
            if not poly.is_valid:
                poly = make_valid(poly)
            arr_clean = geom_to_single_ring(poly, w, h, simplify_tol=simplify_tol)
            if arr_clean is None:
                return None
            arr = arr_clean
        except Exception:  # noqa: BLE001
            return None

    if len(arr) < 3 or poly_area_at_640(arr, w, h) < min_area_640:
        return None
    return arr


# Alias used by Scripts #2 and #4
clean_polygon = clean_and_normalize_polygon


def _union_polygons(polys: list[np.ndarray], w: int, h: int) -> np.ndarray | None:
    if not polys:
        return None
    if len(polys) == 1:
        return polys[0]
    if HAS_SHAPELY:
        geoms = []
        for p in polys:
            g = Polygon(p)
            if not g.is_valid:
                g = make_valid(g)
            geoms.append(g)
        merged = unary_union(geoms)
        return geom_to_single_ring(merged, w, h)
    ring = polys[0]
    for p in polys[1:]:
        ring = _bridge_two_rings(ring, p)
    return np.clip(ring, 0.0, 1.0)


# --------------------------------------------------------------------------- Rider-Vehicle Union
def _match_score_explicit_rider(
    rider_box: tuple[float, float, float, float],
    veh_box: tuple[float, float, float, float],
) -> float:
    vx1, vy1, vx2, vy2 = veh_box
    vw, vh = max(1e-6, vx2 - vx1), max(1e-6, vy2 - vy1)
    exp_veh = (vx1 - 0.18 * vw, vy1 - 0.45 * vh, vx2 + 0.18 * vw, vy2 + 0.10 * vh)
    ios = bbox_ios(rider_box, exp_veh)
    if ios < 0.25:
        return 0.0
    rcx = 0.5 * (rider_box[0] + rider_box[2])
    vcx = 0.5 * (vx1 + vx2)
    h_dist = abs(rcx - vcx) / vw
    if h_dist > 0.85:
        return 0.0
    return ios + 0.5 * bbox_iou(rider_box, veh_box) - 0.2 * h_dist


def _match_score_generic_person(
    person_box: tuple[float, float, float, float],
    veh_box: tuple[float, float, float, float],
) -> float:
    px1, py1, px2, py2 = person_box
    vx1, vy1, vx2, vy2 = veh_box
    ph = max(1e-6, py2 - py1)
    vw, vh = max(1e-6, vx2 - vx1), max(1e-6, vy2 - vy1)

    pcx = 0.5 * (px1 + px2)
    if not (vx1 - 0.18 * vw <= pcx <= vx2 + 0.18 * vw):
        return 0.0
    if py1 > vy1 + 0.25 * vh:
        return 0.0
    if not (vy1 + 0.15 * vh <= py2 <= vy2 + 0.15 * vh):
        return 0.0
    h_ratio = ph / vh
    if not (0.35 <= h_ratio <= 2.5):
        return 0.0
    iou = bbox_iou(person_box, veh_box)
    ios = bbox_ios(person_box, veh_box)
    if iou < 0.10 and ios < 0.20:
        return 0.0
    return ios + iou


def _unpack_item(it: Instance | dict[str, Any]) -> tuple[int | str, np.ndarray, float, str]:
    if isinstance(it, Instance):
        role = it.role if it.role is not None else it.class_id
        return role, it.polygon, float(it.score), it.source_tag
    role = it.get("role", it.get("class_id"))
    poly = it.get("poly", it.get("polygon"))
    return role, poly, float(it.get("score", 1.0)), str(it.get("source_tag", ""))


def _pack_item(as_instance: bool, class_id: int, poly: np.ndarray, score: float, source_tag: str = "") -> Any:
    if as_instance:
        return Instance(class_id=class_id, polygon=poly, score=score, source_tag=source_tag)
    return {"class_id": class_id, "poly": poly, "score": score}


def merge_riders_and_vehicles(
    raw_items: list[Any],
    w: int = 1280,
    h: int = 720,
    mode: str = "explicit_rider",
    min_area_640: float = MIN_GT_AREA_PX,
) -> list[Any]:
    """Merge seated riders with motorcycles (ID 5) / bicycles (ID 7) into ONE single polygon instance.

    Accepts either `list[Instance]` or `list[dict]` and returns the same item type.
    """
    if not raw_items:
        return []
    as_instance = isinstance(raw_items[0], Instance)

    vehicles: list[dict[str, Any]] = []
    riders: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []

    for it in raw_items:
        role, poly, score, tag = _unpack_item(it)
        if role in (4, 6):
            vehicles.append({"role": int(role), "poly": poly, "box": poly_bbox(poly), "score": score, "tag": tag, "riders": []})
        elif mode == "explicit_rider" and role in ("rider", "motorcyclist_rider", "bicyclist_rider"):
            riders.append({"role": str(role), "poly": poly, "box": poly_bbox(poly), "score": score, "tag": tag})
        elif mode == "generic_person" and role == 8:
            riders.append({"role": "generic_person", "poly": poly, "box": poly_bbox(poly), "score": score, "tag": tag})
        elif isinstance(role, int) and 0 <= role < len(CLASSES):
            others.append({"class_id": role, "poly": poly, "score": score, "tag": tag})

    unmatched_riders: list[dict[str, Any]] = []
    for r in riders:
        best_v_idx = -1
        best_score = 0.0
        for vi, v in enumerate(vehicles):
            if r["role"] == "motorcyclist_rider" and v["role"] != 4:
                continue
            if r["role"] == "bicyclist_rider" and v["role"] != 6:
                continue
            sc = (
                _match_score_explicit_rider(r["box"], v["box"])
                if mode == "explicit_rider"
                else _match_score_generic_person(r["box"], v["box"])
            )
            if sc > best_score:
                best_score = sc
                best_v_idx = vi
        if best_v_idx >= 0:
            vehicles[best_v_idx]["riders"].append(r)
        else:
            unmatched_riders.append(r)

    out: list[Any] = [_pack_item(as_instance, o["class_id"], o["poly"], o["score"], o["tag"]) for o in others]

    for v in vehicles:
        if v["riders"]:
            merged_cls = 5 if v["role"] == 4 else 7
            polys = [v["poly"]] + [r["poly"] for r in v["riders"]]
            merged_poly = _union_polygons(polys, w, h)
            if merged_poly is not None and poly_area_at_640(merged_poly, w, h) >= min_area_640:
                best_sc = max(v["score"], max(r["score"] for r in v["riders"]))
                out.append(_pack_item(as_instance, merged_cls, merged_poly, best_sc, v["tag"]))
        else:
            out.append(_pack_item(as_instance, v["role"], v["poly"], v["score"], v["tag"]))

    for r in unmatched_riders:
        if r["role"] == "motorcyclist_rider":
            cid = 5
        elif r["role"] == "bicyclist_rider":
            cid = 7
        else:
            cid = 8
        out.append(_pack_item(as_instance, cid, r["poly"], r["score"], r["tag"]))

    return out


# --------------------------------------------------------------------------- Z-Order Non-Overlap
def enforce_non_overlap(
    items: list[Any],
    w: int,
    h: int,
    min_area_640: float = MIN_GT_AREA_PX,
    same_class_dedup_iou: float = 0.65,
) -> list[Any]:
    """Enforce Competition Rule #2: front object keeps pixels, behind object keeps visible remainder."""
    if len(items) <= 1:
        return items
    as_instance = isinstance(items[0], Instance)

    enriched = []
    for it in items:
        role, poly, score, tag = _unpack_item(it)
        b = poly_bbox(poly)
        z_key = (b[3], poly_area_norm(poly), score)
        enriched.append((z_key, b, int(role), poly, score, tag))
    enriched.sort(key=lambda x: x[0], reverse=True)

    kept: list[Any] = []
    kept_cls: list[int] = []
    kept_boxes: list[tuple[float, float, float, float]] = []
    accum_front = None

    for _, box, cid, poly_pts, score, tag in enriched:
        if any(kc == cid and bbox_iou(box, kb) >= same_class_dedup_iou for kc, kb in zip(kept_cls, kept_boxes)):
            continue

        if not HAS_SHAPELY:
            kept.append(_pack_item(as_instance, cid, poly_pts, score, tag))
            kept_cls.append(cid)
            kept_boxes.append(box)
            continue

        geom = Polygon(poly_pts)
        if not geom.is_valid:
            geom = make_valid(geom)
        if geom.is_empty:
            continue

        if accum_front is not None:
            visible = geom.difference(accum_front)
            ring = geom_to_single_ring(visible, w, h)
            if ring is None or poly_area_at_640(ring, w, h) < min_area_640:
                continue
            final_poly = ring
            final_geom = Polygon(ring)
            if not final_geom.is_valid:
                final_geom = make_valid(final_geom)
        else:
            final_poly = poly_pts
            final_geom = geom

        kept.append(_pack_item(as_instance, cid, final_poly, score, tag))
        kept_cls.append(cid)
        kept_boxes.append(poly_bbox(final_poly))
        accum_front = final_geom if accum_front is None else unary_union([accum_front, final_geom])

    return kept


# --------------------------------------------------------------------------- YOLO-seg I/O & Drawing
def read_yolo_seg_txt(txt_path: Path | None) -> tuple[list[tuple[int, np.ndarray]], list[str]]:
    if txt_path is None or not txt_path.is_file():
        return [], []
    items: list[tuple[int, np.ndarray]] = []
    errors: list[str] = []
    text = txt_path.read_text(encoding="utf-8", errors="replace")
    for lineno, raw in enumerate(text.splitlines(), start=1):
        s = raw.strip()
        if not s:
            continue
        parts = s.split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            errors.append(f"{txt_path.name}:{lineno}: expected >=7 tokens (class + >=3 xy pairs), got {len(parts)}")
            continue
        try:
            cid = int(parts[0])
            coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float32).reshape(-1, 2)
        except ValueError as exc:
            errors.append(f"{txt_path.name}:{lineno}: parse error ({exc})")
            continue
        if not (0 <= cid < len(CLASSES)):
            errors.append(f"{txt_path.name}:{lineno}: class_id {cid} out of range 0..{len(CLASSES)-1}")
            continue
        items.append((cid, coords))
    return items, errors


def read_yolo_seg_file(txt_path: Path | None) -> tuple[list[Instance], list[str]]:
    raw_items, errors = read_yolo_seg_txt(txt_path)
    return [Instance(class_id=cid, polygon=pts) for cid, pts in raw_items], errors


def write_yolo_seg_txt(txt_path: Path, items: list[Any]) -> None:
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    for it in items:
        role, pts, _, _ = _unpack_item(it)
        cid = int(role)
        arr = np.asarray(pts, dtype=np.float32)
        if len(arr) < 3:
            continue
        flat = " ".join(f"{float(v):.6f}" for v in np.clip(arr, 0.0, 1.0).flatten())
        lines.append(f"{cid} {flat}")
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


write_yolo_seg_file = write_yolo_seg_txt


def write_data_yaml(yaml_path: Path, train_rel: str = "images/train", val_rel: str = "images/val") -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "path: .",
        f"train: {train_rel}",
        f"val: {val_rel}",
        f"nc: {len(CLASSES)}",
        "names:",
    ]
    for i, name in enumerate(CLASSES):
        lines.append(f"  {i}: {name}")
    yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def draw_polygons_on_image(
    img: Image.Image,
    items: list[Any],
    fill_alpha: int = 75,
) -> Image.Image:
    base = img.convert("RGBA")
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    w, h = base.size
    for it in items:
        if isinstance(it, tuple):
            cid, pts = it
        else:
            cid, pts, _, _ = _unpack_item(it)
        xy = [(float(x) * w, float(y) * h) for x, y in pts]
        if len(xy) < 3:
            continue
        col = CLASS_COLORS[int(cid) % len(CLASS_COLORS)]
        draw.polygon(xy, fill=col + (fill_alpha,), outline=col + (255,), width=2)
        x0, y0 = min(p[0] for p in xy), min(p[1] for p in xy)
        draw.text((x0 + 2, max(0.0, y0 - 11)), CLASSES[int(cid)], fill=col + (255,))
    return Image.alpha_composite(base, overlay).convert("RGB")


def render_contact_sheet(
    pairs: list[tuple[Path, list[Any]]],
    out_path: Path,
    max_side: int = 420,
    cols: int = 4,
) -> Path | None:
    """Save a visual grid of images overlaid with their YOLO-seg polygons for quick human QC."""
    if not pairs:
        return None
    rows_n = (len(pairs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * max_side, rows_n * max_side), (25, 25, 25))
    for k, (img_path, insts) in enumerate(pairs):
        with Image.open(img_path) as im:
            vis = draw_polygons_on_image(im.convert("RGB"), insts)
        vis.thumbnail((max_side, max_side))
        sheet.paste(vis, ((k % cols) * max_side, (k // cols) * max_side))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(out_path, quality=88)
    return out_path
