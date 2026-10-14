"""Per-sequence statistics for MOT17 (results/data_stats.csv)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from cvdt.data import filtering
from cvdt.data.config import dataset_config
from cvdt.data.filtering import Category
from cvdt.data.mot import MOTClass, load_gt, read_seqinfo

HEIGHT_BUCKETS = {"small": (0, 50), "medium": (50, 150), "large": (150, np.inf)}  # px, [lo, hi)
VIS_BUCKETS = {
    "<0.25": (0, 0.25),
    "0.25-0.5": (0.25, 0.5),
    "0.5-0.75": (0.5, 0.75),
    ">=0.75": (0.75, 1.01),
}


def camera_motion(img_dir: Path, n_frames: int, step: int = 10, work_width: int = 640) -> dict:
    """Estimate global (background) motion between consecutive frames.

    For frame pairs (t, t+1) sampled every ``step`` frames, fit a similarity
    transform to ORB matches with RANSAC (moving people are outliers) and
    record the translation magnitude, rescaled to native pixels per frame.
    A static camera gives ~0 px; a moving one gives a clearly positive median.
    """
    orb = cv2.ORB_create(2000)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    shifts = []
    for t in range(1, n_frames, step):
        a = cv2.imread(str(img_dir / f"{t:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        b = cv2.imread(str(img_dir / f"{t + 1:06d}.jpg"), cv2.IMREAD_GRAYSCALE)
        scale = work_width / a.shape[1]
        a, b = (cv2.resize(x, None, fx=scale, fy=scale) for x in (a, b))
        ka, da = orb.detectAndCompute(a, None)
        kb, db = orb.detectAndCompute(b, None)
        if da is None or db is None:
            continue
        m = bf.match(da, db)
        if len(m) < 20:
            continue
        pa = np.float32([ka[x.queryIdx].pt for x in m])
        pb = np.float32([kb[x.trainIdx].pt for x in m])
        M, _ = cv2.estimateAffinePartial2D(pa, pb, method=cv2.RANSAC, ransacReprojThreshold=2.0)
        if M is not None:
            shifts.append(float(np.hypot(M[0, 2], M[1, 2]) / scale))
    s = np.array(shifts)
    return {
        "camera_motion_px_median": round(float(np.median(s)), 2) if len(s) else np.nan,
        "camera_motion_px_p90": round(float(np.percentile(s, 90)), 2) if len(s) else np.nan,
    }


def _bucket_counts(values: np.ndarray, buckets: dict) -> dict[str, int]:
    return {k: int(((values >= lo) & (values < hi)).sum()) for k, (lo, hi) in buckets.items()}


def sequence_stats(seq_dir: Path, meta: dict, min_visibility: float) -> dict:
    si = read_seqinfo(seq_dir / "seqinfo.ini")
    n = int(si["seqLength"])
    df = load_gt(seq_dir / "gt" / "gt.txt")
    cat = filtering.categorize(
        df["cls"].to_numpy(), df["conf"].to_numpy(), df["vis"].to_numpy(), min_visibility
    )
    reason = filtering.ignore_reason(df["cls"].to_numpy(), cat)
    ped = df[(df["cls"] == MOTClass.PEDESTRIAN) & (df["conf"] == 1)]
    kept = df[cat == Category.POSITIVE]
    per_frame = ped.groupby("frame").size().reindex(range(1, n + 1), fill_value=0)

    row: dict = {
        "sequence": seq_dir.name,
        "camera": meta["camera"],
        "viewpoint": meta.get("viewpoint", ""),
        "frames": n,
        "width": int(si["imWidth"]),
        "height": int(si["imHeight"]),
        "fps": float(si["frameRate"]),
        "gt_tracks": int(ped["id"].nunique()),
        "gt_ids_all_classes": int(df["id"].nunique()),
        "boxes_all": len(df),
    }
    counts = df["cls"].value_counts()
    for c in MOTClass:
        row[f"boxes_cls{c.value:02d}_{c.name.lower()}"] = int(counts.get(c.value, 0))
    row.update(
        {
            "ped_flag1": len(ped),
            "ped_flag0": int(((df["cls"] == MOTClass.PEDESTRIAN) & (df["conf"] == 0)).sum()),
            "kept_positive": int((cat == Category.POSITIVE).sum()),
            "low_visibility": int((cat == Category.LOW_VISIBILITY).sum()),
            "ignore_distractor_class": int((reason == filtering.REASON_DISTRACTOR).sum()),
            "ignore_zero_flag": int((reason == filtering.REASON_ZERO_FLAG).sum()),
            "dropped_non_person": int((cat == Category.DROPPED).sum()),
            "people_per_frame_mean": round(float(per_frame.mean()), 2),
            "people_per_frame_max": int(per_frame.max()),
            "ped_vis_median": round(float(ped["vis"].median()), 3),
        }
    )
    for k, v in _bucket_counts(ped["vis"].to_numpy(), VIS_BUCKETS).items():
        row[f"ped_vis_{k}_pct"] = round(100 * v / max(len(ped), 1), 1)
    h = kept["h"].to_numpy()
    for k, v in _bucket_counts(h, HEIGHT_BUCKETS).items():
        row[f"kept_h_{k}"] = v
        row[f"kept_h_{k}_pct"] = round(100 * v / max(len(h), 1), 1)
    row["kept_h_median_px"] = round(float(np.median(h)), 1) if len(h) else np.nan
    row.update(camera_motion(seq_dir / si.get("imDir", "img1"), n))
    return row


def compute_stats(canon_root: Path, min_visibility: float) -> pd.DataFrame:
    meta = dataset_config()["sequences"]
    seqs = sorted(p for p in canon_root.iterdir() if p.is_dir() and p.name in meta)
    df = pd.DataFrame([sequence_stats(p, meta[p.name], min_visibility) for p in seqs])
    df["share_of_kept_pct"] = (100 * df["kept_positive"] / df["kept_positive"].sum()).round(1)
    df["share_of_ped_flag1_pct"] = (100 * df["ped_flag1"] / df["ped_flag1"].sum()).round(1)
    return df


MD_COLUMNS = {
    "sequence": "Seq",
    "camera": "Camera",
    "camera_motion_px_median": "Motion px/f (med)",
    "frames": "Frames",
    "width": "W",
    "height": "H",
    "fps": "FPS",
    "gt_tracks": "GT tracks",
    "ped_flag1": "Ped (flag 1)",
    "kept_positive": "Kept",
    "share_of_kept_pct": "Kept %",
    "low_visibility": "Low-vis",
    "ignore_distractor_class": "Ign. distr.",
    "ignore_zero_flag": "Ign. flag0",
    "people_per_frame_mean": "People/f mean",
    "people_per_frame_max": "People/f max",
    "ped_vis_<0.25_pct": "Vis<0.25 %",
    "ped_vis_median": "Vis med",
    "kept_h_small_pct": "H<50 %",
    "kept_h_medium_pct": "H 50-150 %",
    "kept_h_large_pct": "H>150 %",
    "kept_h_median_px": "H med px",
}


def to_markdown(df: pd.DataFrame) -> str:
    t = df[list(MD_COLUMNS)].rename(columns=MD_COLUMNS)
    lines = ["| " + " | ".join(t.columns) + " |", "|" + "---|" * len(t.columns)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in t.itertuples(index=False)]
    return "\n".join(lines)
