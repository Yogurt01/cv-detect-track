"""Label filtering rule for MOT17 detection training and detection evaluation.

Every ground-truth box falls into exactly one :class:`Category`:

``POSITIVE``
    class ``PEDESTRIAN`` with ``conf_flag == 1`` and ``visibility >= min_visibility``.
    Becomes a YOLO training label (single class "pedestrian").

``LOW_VISIBILITY``
    class ``PEDESTRIAN`` with ``conf_flag == 1`` but ``visibility < min_visibility``
    (default 0.25, set in ``configs/data/mot17.yaml``). Not a training label.
    Written to the ignore sidecar with reason ``low_visibility`` so detection
    evaluation can treat it as an ignore region rather than as background.

``IGNORE``
    Person-like classes that MOT evaluation neither rewards nor penalises:
    ``PERSON_ON_VEHICLE``, ``STATIC_PERSON``, ``DISTRACTOR``, ``REFLECTION``
    (MOT16 paper Sec. 4.1.1; TrackEval's distractor list for MOT17), regardless
    of their flag; plus ``PEDESTRIAN`` with ``conf_flag == 0``. Written to the
    ignore sidecar with reason ``distractor_class`` or ``zero_flag``.

``DROPPED``
    Non-person classes, used by the annotators only to compute occlusion:
    ``CAR``, ``BICYCLE``, ``MOTORBIKE``, ``NON_MOTORIZED_VEHICLE``, ``OCCLUDER``,
    ``OCCLUDER_ON_GROUND``, ``OCCLUDER_FULL``. Not used anywhere downstream.

The visibility threshold applies to training labels only. Tracking evaluation
reads the raw ``gt.txt`` and applies TrackEval's standard MOT17 handling (no
visibility filter), so nothing here changes tracking ground truth.
"""

from __future__ import annotations

from enum import IntEnum

import numpy as np

from cvdt.data.mot import MOTClass

DEFAULT_MIN_VISIBILITY = 0.25

POSITIVE_CLASSES = frozenset({MOTClass.PEDESTRIAN})
IGNORE_CLASSES = frozenset(
    {MOTClass.PERSON_ON_VEHICLE, MOTClass.STATIC_PERSON, MOTClass.DISTRACTOR, MOTClass.REFLECTION}
)
DROPPED_CLASSES = frozenset(
    {
        MOTClass.CAR,
        MOTClass.BICYCLE,
        MOTClass.MOTORBIKE,
        MOTClass.NON_MOTORIZED_VEHICLE,
        MOTClass.OCCLUDER,
        MOTClass.OCCLUDER_ON_GROUND,
        MOTClass.OCCLUDER_FULL,
    }
)
assert not (POSITIVE_CLASSES & IGNORE_CLASSES or IGNORE_CLASSES & DROPPED_CLASSES)
assert len(POSITIVE_CLASSES | IGNORE_CLASSES | DROPPED_CLASSES) == len(MOTClass)


class Category(IntEnum):
    POSITIVE = 0
    LOW_VISIBILITY = 1
    IGNORE = 2
    DROPPED = 3


# Reason strings stored in the ignore sidecar.
REASON_LOW_VIS = "low_visibility"
REASON_DISTRACTOR = "distractor_class"
REASON_ZERO_FLAG = "zero_flag"


def categorize(
    classes: np.ndarray,
    conf_flags: np.ndarray,
    visibility: np.ndarray,
    min_visibility: float = DEFAULT_MIN_VISIBILITY,
) -> np.ndarray:
    """Return a :class:`Category` value per box (int array, same length as inputs)."""
    classes = np.asarray(classes)
    conf_flags = np.asarray(conf_flags)
    visibility = np.asarray(visibility)
    unknown = ~np.isin(classes, [int(c) for c in MOTClass])
    if unknown.any():
        raise ValueError(f"unknown MOT class ids: {sorted(set(classes[unknown].tolist()))}")

    out = np.full(classes.shape, int(Category.DROPPED), dtype=np.int64)
    is_ped = np.isin(classes, [int(c) for c in POSITIVE_CLASSES])
    out[np.isin(classes, [int(c) for c in IGNORE_CLASSES])] = Category.IGNORE
    out[is_ped & (conf_flags == 0)] = Category.IGNORE
    active = is_ped & (conf_flags == 1)
    out[active & (visibility >= min_visibility)] = Category.POSITIVE
    out[active & (visibility < min_visibility)] = Category.LOW_VISIBILITY
    return out


def ignore_reason(classes: np.ndarray, categories: np.ndarray) -> np.ndarray:
    """Sidecar reason string per box ('' for POSITIVE and DROPPED)."""
    reasons = np.full(len(classes), "", dtype=object)
    reasons[categories == Category.LOW_VISIBILITY] = REASON_LOW_VIS
    ign = categories == Category.IGNORE
    reasons[ign & np.isin(classes, [int(c) for c in IGNORE_CLASSES])] = REASON_DISTRACTOR
    reasons[ign & np.isin(classes, [int(c) for c in POSITIVE_CLASSES])] = REASON_ZERO_FLAG
    return reasons
