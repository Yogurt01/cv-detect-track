"""Detection evaluation with ignore regions (DECISIONS.md D10, D13, D18, D19).

Ground truth (read from ``gt.txt``, never from the training sidecar):

* ``primary`` protocol: pedestrians (class 1, flag 1) with visibility >=
  ``ignore_visibility_below`` are positives; pedestrians below it and the
  distractor classes (2, 7, 8, 12) are ignore regions. The threshold comes from
  ``eval.detection.ignore_visibility_below`` in ``configs/data/mot17.yaml``.
* ``secondary`` protocol: all pedestrians are positives; distractor classes stay
  ignore regions (D18).
* Non-person classes are not used. GT boxes are clipped to the image (D18).

Matching per image (D13): detections sorted by score (descending, stable);
each is matched one-to-one to the unmatched positive with the highest IoU if
that IoU >= the threshold (the pycocotools rule, including its tie handling).
Unmatched positives are FN. A detection left unmatched is then handled by the
ignore rule (all overlap tests use 0.5, at every AP threshold):

``ignore_rule="d19"`` (default, D19), in this order:
  1. duplicate guard: IoU >= 0.5 with any positive (matched or not) -> FP,
     never excused by an ignore region;
  2. IoU >= 0.5 with a distractor-class ignore box -> removed;
  3. IoA >= 0.5 with a low-visibility pedestrian ignore box -> removed, where
     IoA = area(det & box) / area(det);
  4. otherwise FP.

``ignore_rule="d13"`` (sensitivity analysis): IoU >= 0.5 with any ignore box
(distractor or low-visibility) -> removed, otherwise FP.

AP: COCO 101-point interpolation over all images pooled, with the same
ordering and precision envelope as ``pycocotools.cocoeval.COCOeval.accumulate``
and no per-image detection cap. AP50 and AP50:95 (10 thresholds) are
reported, plus precision/recall at a fixed operating confidence (0.25, D15).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from cvdt.data import filtering
from cvdt.data.filtering import Category
from cvdt.data.mot import load_gt, mot_to_xyxy, read_seqinfo
from cvdt.data.yolo import clip_xyxy

IOU_THRESHOLDS = np.linspace(0.5, 0.95, 10)
RECALL_THRESHOLDS = np.linspace(0.0, 1.0, 101)
IGNORE_IOU = 0.5  # distractor ignore regions and the duplicate guard (IoU)
IGNORE_IOA = 0.5  # low-visibility ignore regions under D19 (IoA)
OPERATING_CONF = 0.25
IGNORE_RULES = ("d19", "d13")

TP, FP, IGNORED = 1, 0, -1

# Per-detection reasons (diagnostics), returned by match_image(..., return_reasons=True).
R_TP = 0
R_FP = 1  # unmatched, no ignore region applies
R_FP_GUARD = 2  # D19 duplicate guard fired, no ignore region would have excused it
R_FP_GUARD_EXCUSABLE = 3  # D19 duplicate guard fired although an ignore region applied
R_IGN_DISTRACTOR = 4
R_IGN_LOWVIS = 5


def _empty() -> np.ndarray:
    return np.zeros((0, 4))


@dataclass(frozen=True)
class ImageGT:
    """Evaluation GT of one image, xyxy boxes.

    ``distractors``: ignore boxes of classes 2, 7, 8, 12. ``low_vis``: ignore
    boxes of pedestrians below the evaluation visibility threshold (primary
    protocol only).
    """

    positives: np.ndarray
    distractors: np.ndarray = field(default_factory=_empty)
    low_vis: np.ndarray = field(default_factory=_empty)


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU between xyxy boxes ``a`` (N, 4) and ``b`` (M, 4); continuous coordinates."""
    a = np.asarray(a, dtype=np.float64).reshape(-1, 4)
    b = np.asarray(b, dtype=np.float64).reshape(-1, 4)
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    iw = np.clip(
        np.minimum(a[:, None, 2], b[None, :, 2]) - np.maximum(a[:, None, 0], b[None, :, 0]), 0, None
    )
    ih = np.clip(
        np.minimum(a[:, None, 3], b[None, :, 3]) - np.maximum(a[:, None, 1], b[None, :, 1]), 0, None
    )
    inter = iw * ih
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = area_a[:, None] + area_b[None, :] - inter
    return np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)


def load_eval_gt(
    seq_dir: Path,
    ignore_visibility_below: float,
    protocol: str = "primary",
    clip: bool = True,
) -> dict[int, ImageGT]:
    """Per-frame evaluation GT for one canonical MOT17 sequence."""
    if protocol not in ("primary", "secondary"):
        raise ValueError(f"unknown protocol {protocol!r}")
    si = read_seqinfo(seq_dir / "seqinfo.ini")
    w, h, n = int(si["imWidth"]), int(si["imHeight"]), int(si["seqLength"])
    df = load_gt(seq_dir / "gt" / "gt.txt")
    cat = filtering.categorize(
        df["cls"].to_numpy(), df["conf"].to_numpy(), df["vis"].to_numpy(), ignore_visibility_below
    )
    boxes = mot_to_xyxy(df[["x", "y", "w", "h"]].to_numpy())
    if clip:
        boxes = clip_xyxy(boxes, w, h)
    pos = cat == Category.POSITIVE
    distractor = cat == Category.IGNORE  # MOT17 has no pedestrian with flag 0 (data stage)
    low = cat == Category.LOW_VISIBILITY
    if protocol == "secondary":
        pos, low = pos | low, np.zeros_like(low)
    frames = df["frame"].to_numpy()
    return {
        f: ImageGT(
            positives=boxes[pos & (frames == f)],
            distractors=boxes[distractor & (frames == f)],
            low_vis=boxes[low & (frames == f)],
        )
        for f in range(1, n + 1)
    }


def ioa_matrix(det: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """Intersection over detection area between ``det`` (N, 4) and ``boxes`` (M, 4)."""
    det = np.asarray(det, dtype=np.float64).reshape(-1, 4)
    boxes = np.asarray(boxes, dtype=np.float64).reshape(-1, 4)
    if len(det) == 0 or len(boxes) == 0:
        return np.zeros((len(det), len(boxes)))
    iw = np.clip(
        np.minimum(det[:, None, 2], boxes[None, :, 2])
        - np.maximum(det[:, None, 0], boxes[None, :, 0]),
        0,
        None,
    )
    ih = np.clip(
        np.minimum(det[:, None, 3], boxes[None, :, 3])
        - np.maximum(det[:, None, 1], boxes[None, :, 1]),
        0,
        None,
    )
    area = ((det[:, 2] - det[:, 0]) * (det[:, 3] - det[:, 1]))[:, None]
    return np.where(area > 0, iw * ih / np.where(area > 0, area, 1), 0.0)


def _max_or_zero(m: np.ndarray) -> np.ndarray:
    return m.max(axis=1) if m.shape[1] else np.zeros(m.shape[0])


def match_image(
    det_boxes: np.ndarray,
    det_scores: np.ndarray,
    gt: ImageGT,
    iou_thr: float,
    ignore_rule: str = "d19",
    return_reasons: bool = False,
):
    """Label each detection TP / FP / IGNORED at one IoU threshold.

    Returns ``(scores, labels)`` in matching order (score descending, stable),
    plus a per-detection reason code array (``R_*``) if ``return_reasons``.
    """
    if ignore_rule not in IGNORE_RULES:
        raise ValueError(f"unknown ignore rule {ignore_rule!r}; choose from {IGNORE_RULES}")
    order = np.argsort(-np.asarray(det_scores), kind="mergesort")
    scores = np.asarray(det_scores)[order]
    boxes = np.asarray(det_boxes).reshape(-1, 4)[order]
    labels = np.full(len(order), FP, dtype=np.int64)
    ious = iou_matrix(boxes, gt.positives)
    matched = np.zeros(len(gt.positives), dtype=bool)
    for d in range(len(boxes)):
        best, m = min(iou_thr, 1 - 1e-10), -1
        for g in range(len(gt.positives)):
            if matched[g] or ious[d, g] < best:
                continue
            best, m = ious[d, g], g
        if m >= 0:
            matched[m] = True
            labels[d] = TP
    reasons = np.where(labels == TP, R_TP, R_FP)
    idx = np.flatnonzero(labels == FP)
    if len(idx):
        ub = boxes[idx]
        on_distractor = _max_or_zero(iou_matrix(ub, gt.distractors)) >= IGNORE_IOU
        if ignore_rule == "d13":
            on_low = _max_or_zero(iou_matrix(ub, gt.low_vis)) >= IGNORE_IOU
            excused = on_distractor | on_low
            reasons[idx[on_low]] = R_IGN_LOWVIS
            reasons[idx[on_distractor]] = R_IGN_DISTRACTOR
        else:
            on_low = _max_or_zero(ioa_matrix(ub, gt.low_vis)) >= IGNORE_IOA
            guard = _max_or_zero(ious[idx]) >= IGNORE_IOU
            excusable = on_distractor | on_low
            excused = ~guard & excusable
            reasons[idx[~guard & on_low]] = R_IGN_LOWVIS
            reasons[idx[~guard & on_distractor]] = R_IGN_DISTRACTOR
            reasons[idx[guard & ~excusable]] = R_FP_GUARD
            reasons[idx[guard & excusable]] = R_FP_GUARD_EXCUSABLE
        labels[idx[excused]] = IGNORED
    if return_reasons:
        return scores, labels, reasons
    return scores, labels


def average_precision(scores: np.ndarray, labels: np.ndarray, n_pos: int) -> float:
    """COCO 101-point AP from pooled detections (pycocotools ``accumulate`` semantics).

    ``scores`` / ``labels`` must be the per-image matching outputs concatenated in
    image order; they are re-sorted stably by score here, as pycocotools does.
    """
    if n_pos == 0:
        return float("nan")
    keep = labels != IGNORED
    order = np.argsort(-scores, kind="mergesort")
    lab = labels[order][keep[order]]
    tp = np.cumsum(lab == TP).astype(float)
    fp = np.cumsum(lab == FP).astype(float)
    rc = tp / n_pos
    pr = (tp / (fp + tp + np.spacing(1))).tolist()
    for i in range(len(pr) - 1, 0, -1):
        if pr[i] > pr[i - 1]:
            pr[i - 1] = pr[i]
    q = np.zeros(len(RECALL_THRESHOLDS))
    inds = np.searchsorted(rc, RECALL_THRESHOLDS, side="left")
    for ri, pi in enumerate(inds):
        if pi < len(pr):
            q[ri] = pr[pi]
    return float(q.mean())


def evaluate_images(
    dets: dict[int, tuple[np.ndarray, np.ndarray]],
    gts: dict[int, ImageGT],
    operating_conf: float = OPERATING_CONF,
    ignore_rule: str = "d19",
) -> dict[str, float]:
    """Evaluate ``{image: (boxes, scores)}`` against ``{image: ImageGT}``.

    Images missing from ``dets`` have no detections. Returns AP50, AP50_95 and
    TP / FP / FN / ignored counts, precision / recall and per-reason counts at
    ``operating_conf`` (IoU 0.5).
    """
    images = sorted(gts)
    n_pos = int(sum(len(gts[i].positives) for i in images))
    empty = (np.zeros((0, 4)), np.zeros(0))
    aps = []
    op = {}
    for t in IOU_THRESHOLDS:
        all_s, all_l, all_r = [], [], []
        for i in images:
            b, s = dets.get(i, empty)
            sc, lab, rsn = match_image(b, s, gts[i], float(t), ignore_rule, return_reasons=True)
            all_s.append(sc)
            all_l.append(lab)
            all_r.append(rsn)
        s_cat = np.concatenate(all_s) if all_s else np.zeros(0)
        l_cat = np.concatenate(all_l) if all_l else np.zeros(0, dtype=np.int64)
        r_cat = np.concatenate(all_r) if all_r else np.zeros(0, dtype=np.int64)
        aps.append(average_precision(s_cat, l_cat, n_pos))
        if np.isclose(t, 0.5):
            sel = s_cat >= operating_conf
            n_tp = int(((l_cat == TP) & sel).sum())
            n_fp = int(((l_cat == FP) & sel).sum())
            op = {
                "TP": n_tp,
                "FP": n_fp,
                "FN": n_pos - n_tp,
                "ignored": int(((l_cat == IGNORED) & sel).sum()),
                "precision": n_tp / (n_tp + n_fp) if n_tp + n_fp else float("nan"),
                "recall": n_tp / n_pos if n_pos else float("nan"),
                "ignored_distractor": int(((r_cat == R_IGN_DISTRACTOR) & sel).sum()),
                "ignored_low_vis": int(((r_cat == R_IGN_LOWVIS) & sel).sum()),
                "fp_duplicate_guard": int(
                    (((r_cat == R_FP_GUARD) | (r_cat == R_FP_GUARD_EXCUSABLE)) & sel).sum()
                ),
                "fp_guard_overrode_ignore": int(((r_cat == R_FP_GUARD_EXCUSABLE) & sel).sum()),
            }
    return {
        "AP50": aps[0],
        "AP50_95": float(np.mean(aps)),
        "n_pos": n_pos,
        "operating_conf": operating_conf,
        "ignore_rule": ignore_rule,
        **op,
    }
