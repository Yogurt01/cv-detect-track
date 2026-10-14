# Decisions

Append-only, ADR-lite. To change a decision, add a new entry that states "Supersedes Dn"; past entries are never edited.

---

## D1 — Phase 1 scope: pedestrian speed–accuracy study on MOT17

- **Date:** 2026-10-14
- **Context:** The project has about 14 working days, a 4 GB laptop GPU and a 30 GPU-h/week cloud quota. It needs a result that holds up whatever the numbers turn out to be.
- **Options considered:**
  - (A) pedestrian speed–accuracy study on MOT17;
  - (B) domain adaptation to small aerial objects (drone footage);
  - (C) tracking under uniform appearance (dancers, sports).
- **Decision:** Phase 1 is option A on MOT17, with all results stratified by static vs moving camera. The code stays dataset-agnostic, so a Phase 2 can add a small-object / drone dataset (e.g. VisDrone, edge-AI direction) without refactoring.
- **Rationale:**
  - Tracking ground truth exists, compute needs are light, and the result is robust: a Pareto curve plus failure analysis is a finding even if fine-tuning does not help.
  - Camera motion is the main axis along which MOT17 sequences differ for a tracker.
- **Consequences:**
  - Dataset access goes through `SequenceDataset`.
  - Per-sequence metadata (camera type, viewpoint) lives in config, not code.
  - Every reported table has a static / moving breakdown.

## D2 — Two-tier evaluation protocol

- **Date:** 2026-10-14
- **Context:** MOT17 has only 7 unique train sequences, and its test ground truth is hidden. Anything tuned on a sequence cannot be evaluated fairly on that same sequence.
- **Options considered:**
  - the literature "half" split as headline;
  - a single sequence-level train/val/test split for everything;
  - two tiers separating zero-training results from learned results.
- **Decision:**
  - **Tier 1, zero training.** COCO-pretrained detectors, default tracker parameters, thresholds fixed in advance. Evaluated on all 7 unique MOT17 train sequences and reported per sequence and per camera type.
  - **Tier 2, anything learned from MOT17.** Fine-tuning, tracker tuning and threshold selection use a sequence-level held-out split (train / val / test), where test = 1 static + 1 moving sequence.
  - **Selection-leakage rule.** Tier 2 decisions may use only train/val sequences, never Tier 1 results on Tier 2 test sequences.
  - The literature "half" split may be reported later only as a secondary number labelled "leaky, for comparability only".
- **Rationale:**
  - Tier 1 uses all data, because nothing is learned from it.
  - Tier 2 keeps a clean held-out estimate for learned choices.
  - The half split leaks scene and identity information.
- **Consequences:**
  - Split configs live in `configs/data/`.
  - A leakage test runs on every split config.
  - The Tier 2 split is chosen from candidates before any Tier 2 work starts.
  - With only 2 test sequences, per-sequence numbers matter more than means.

## D3 — Own ByteTrack-style tracker is a core deliverable

- **Date:** 2026-10-14
- **Context:** Library-only usage says little about understanding of tracking.
- **Options considered:**
  - own tracker as a core deliverable;
  - own tracker as a stretch goal;
  - library trackers only.
- **Decision:**
  - Implement a ByteTrack-style tracker (Kalman filter, IoU cost, Hungarian matching, two-stage high/low-confidence association) as a core, timeboxed deliverable in a later stage.
  - Parity with the library implementation is a documented comparison, not a pass/fail target.
  - The deliverable includes a stage-2 (low-confidence association) ablation.
- **Rationale:** Owning the tracker makes the association ablation and the failure analysis possible, and the timebox bounds the risk.
- **Consequences:**
  - The tracker implements the `Tracker` interface.
  - Kalman and matching unit tests come with that stage.
  - Pre-registered hypothesis H1 in `EXPERIMENT_LOG.md` targets the stage-2 ablation.

## D4 — Setup and data stage first, then stop for review

- **Date:** 2026-10-14
- **Context:** Data handling (deduplication, filtering, splits) is the main leakage risk and should be reviewed before anything is built on it.
- **Options considered:**
  - run setup, data and baselines in one go;
  - stop after the data stage.
- **Decision:** Setup and data stage run first. Work stops after the data stage until the next stage is approved.
- **Rationale:** Mistakes in labels or splits invalidate every later number.
- **Consequences:** No inference pipeline, detector runs or training until approval.

## D5 — AGPL-3.0 license and swappable components

- **Date:** 2026-10-14
- **Context:** Ultralytics, including its YOLO weights, is AGPL-3.0.
- **Options considered:**
  - AGPL-3.0 repository with Ultralytics;
  - permissive repository with a differently licensed detector;
  - permissive repository with Ultralytics (not allowed).
- **Decision:**
  - The repository is AGPL-3.0.
  - Detector, tracker and dataset access sit behind interfaces, so the detector can later be swapped (e.g. a commercially licensed model or an exported ONNX/TensorRT engine) without touching tracking or evaluation code.
- **Rationale:** AGPL is compatible with a public portfolio repository. The interfaces keep a later licensing change cheap.
- **Consequences:**
  - `ultralytics` may be imported only inside `src/cvdt/detect/`, enforced by `tests/test_import_boundary.py`.
  - Media made from MOT17 frames carries the dataset license separately (see README, Licenses & data).

## D6 — Toolchain pins

- **Date:** 2026-10-14
- **Status:** Proposed during implementation; approved by the project owner on 2026-10-14.
- **Context:**
  - Local GPU: RTX 3050 Laptop (Ampere, sm_86, 4 GB), NVIDIA driver 580.178.04, which supports CUDA ≤ 13.0.
  - The Ultralytics version must be ≥ 8.4.63, the release that added OC-SORT, Deep OC-SORT, FastTracker and TrackTrack.
- **Options considered:**
  - Python 3.11 / 3.12 / 3.13;
  - torch CUDA wheels cu126 / cu130;
  - latest vs slightly older torch.
- **Decision:** Python 3.12; `torch==2.14.1` and `torchvision==0.29.1` from the PyTorch cu130 index; `ultralytics==8.4.177`.
- **Rationale:**
  - Python 3.12 is the system Python and has wheels for the whole stack.
  - torch 2.14.1 (2026-09-30) is the latest patch release. Its cu130 build matches the driver's CUDA 13.0 and supports sm_86.
  - The CUDA 13 toolkit ships inside the wheel, so no system toolkit is needed.
  - `ultralytics==8.4.177` was the latest release at setup time and ships all six tracker configs (ByteTrack, BoT-SORT, OC-SORT, Deep OC-SORT, FastTracker, TrackTrack).
- **Consequences:**
  - The driver must support CUDA 13.0. The cu126 index is the fallback for older drivers.
  - Kaggle runs use Kaggle's preinstalled torch and record their own versions.

## D7 — Upstream TrackEval with a NumPy alias shim

- **Date:** 2026-10-14
- **Status:** Proposed during implementation; approved by the project owner on 2026-10-14.
- **Context:**
  - Upstream TrackEval, the official MOTChallenge evaluator (MIT), has had no commits since 2022-11-29.
  - It uses `np.float`, `np.int` and `np.bool`, which NumPy removed in 1.24, and the MOT17 path (IDF1, preprocessing) fails on NumPy 2.x.
  - A maintained PyPI fork exists (`trackeval` 1.3.0); it refactors the HOTA code.
- **Options considered:**
  - pin NumPy < 1.24 (incompatible with the torch / OpenCV stack);
  - use the PyPI fork;
  - use upstream pinned to its last commit plus a shim that restores the removed aliases.
- **Decision:** Upstream TrackEval at commit `12c8791` with `cvdt.eval.trackeval_compat`. The PyPI fork serves as a numerical cross-check only.
- **Rationale:**
  - The removed names were exact aliases of the Python builtins, so restoring them changes no behaviour.
  - Upstream remains the reference implementation.
  - Cross-check on MOT17-02 and MOT17-13 with a perturbed tracker: identical metrics (max abs difference 0).
- **Consequences:**
  - All TrackEval use goes through `import_trackeval()`.
  - `LOG_ON_ERROR` must be disabled in configs; its default writes into the installed package directory.

## D8 — Tier 2 split: candidate C

- **Date:** 2026-10-14
- **Context:** D2 requires a sequence-level Tier 2 split whose test set is 1 static + 1 moving sequence. Three candidates were prepared from the data stage statistics (`configs/data/tier2_candidate_{a,b,c}.yaml`). Small people (< 50 px) are concentrated in MOT17-02, -10 and -13, and MOT17-04 holds 46% of the kept training boxes.
- **Options considered:**
  - A: train 04, 05, 09, 11; val 10; test 02, 13;
  - B: train 02, 05, 09, 11; val 10; test 04, 13;
  - C: train 04, 05, 13; val 09, 11; test 02, 10.
- **Decision:** Candidate C. Train: MOT17-04, -05, -13. Val: MOT17-09, -11. Test: MOT17-02, -10. Recorded as `configs/data/tier2.yaml`.
- **Rationale:**
  - Small people appear in both train (1,475 boxes) and test (1,400 boxes).
  - Val covers both camera types.
  - A gave train only 192 small boxes while test had 1,986, which would confound any conclusion about fine-tuning and small people.
  - B left only 29.6% of the kept boxes for training.
  - Tier 1 already evaluates MOT17-04, so holding it out is unnecessary.
- **Consequences:**
  - Train has only one static sequence (04).
  - Val contains no small people (see D11).
  - The candidate files stay in the repository for traceability.

## D9 — Training-label visibility threshold is a Tier 2 choice

- **Date:** 2026-10-14
- **Context:** The training-label visibility threshold of 0.25 removes 49.4% of the MOT17-02 pedestrian boxes but only 14–31% elsewhere. The best value is not known in advance.
- **Options considered:**
  - keep 0.25 fixed;
  - lower it to a fixed 0.1;
  - treat the threshold as a Tier 2 choice selected on val;
  - mask (grey out) ignore regions in the training images.
- **Decision:**
  - Fine-tune twice with everything else identical: E4a (`min_vis` 0.1) and E4b (`min_vis` 0.25).
  - Select by HOTA on val (MOT17-09, -11).
  - Pre-declared tie rule: if |ΔHOTA| < 1.0, select 0.25 (the original default, and simpler).
  - E5 is a second seed of the selected configuration.
  - No masking or greying of ignore regions in training images.
- **Rationale:**
  - Selecting on val keeps the choice inside Tier 2 (D2).
  - No masking: in MOT17-02, low-visibility and ignore boxes heavily overlap kept positives (e.g. frame 364), so masking would erase visible positives.
- **Consequences:**
  - Ultralytics training has no ignore concept, so boxes below `min_vis` act as background.
  - Two fine-tuning runs instead of one before E5.
  - Config keys: `labels.min_visibility_train` (conversion default) and `labels.min_visibility_train_options` (0.1, 0.25) in `configs/data/mot17.yaml`.

## D10 — Detection evaluation ground truth, fixed in advance

- **Date:** 2026-10-14
- **Context:** Detection metrics depend on which ground-truth boxes count as positives or ignore regions. Tying that to the training threshold (D9) would let a training choice change the yardstick.
- **Options considered:**
  - reuse the training `min_vis` for evaluation;
  - a fixed evaluation threshold, independent of training;
  - no ignore regions at all.
- **Decision:** Fixed in advance, never tuned, independent of D9:
  - Primary: GT pedestrians with visibility < 0.25 and the distractor classes (2, 7, 8, 12) are ignore regions.
  - Secondary: all pedestrian boxes count as positives.
  - Tracking evaluation: standard TrackEval MOT17 handling, unchanged.
- **Rationale:** E4a and E4b must be compared on identical ground truth. The secondary variant shows how much the ignore rule changes the numbers.
- **Consequences:**
  - The evaluation visibility threshold is a separate config key (`eval.detection.ignore_visibility_below: 0.25` in `configs/data/mot17.yaml`) from the training `min_vis`.

## D11 — Input resolution for fine-tuning chosen from Tier 1 results

- **Date:** 2026-10-14
- **Context:** E4/E5 need an input size (640 or 960). Choosing it on the Tier 2 test sequences would be selection leakage, and val (MOT17-09, -11) contains no small people, so it cannot show a resolution effect on small people.
- **Options considered:**
  - choose on val;
  - choose on Tier 1 results excluding the Tier 2 test sequences;
  - create an extra temporally separated val set from MOT17-13.
- **Decision:**
  - imgsz for E4/E5 is chosen from Tier 1 results (E1 vs E2, COCO-pretrained, no learning from MOT17) on sequences 04, 05, 09, 11 and 13 only. MOT17-02 and -10 (the Tier 2 test set) are excluded from this decision.
  - Pre-declared rule: choose 960 only if HOTA improves by ≥ 1.0 on those sequences AND the 960 configuration still meets the real-time threshold; otherwise choose 640.
  - No extra temporally separated val set is created from MOT17-13.
- **Rationale:**
  - Tier 1 involves no learning from MOT17, and excluding 02 and 10 keeps the Tier 2 test set untouched by the choice.
  - A temporal split of MOT17-13 shares scene and identities, so it is still leaky, and it would remove most small-person training data.
- **Consequences:**
  - Documented assumption: the imgsz effect on the pretrained detector transfers to the fine-tuned one.

## D12 — MOT17 license assumption and no redistribution

- **Date:** 2026-10-14
- **Context:** motchallenge.net is offline and no surviving primary source states the MOT17 license. Secondary sources consistently report CC BY-NC-SA 3.0.
- **Options considered:**
  - treat the license as unknown and stop;
  - assume CC BY-NC-SA 3.0 until confirmed;
  - assume a more permissive license.
- **Decision:**
  - Assume CC BY-NC-SA 3.0, the most restrictive plausible terms and therefore a conservative assumption.
  - The project owner is asking the dataset maintainers for confirmation.
  - Until confirmed, no MOT17 frames, annotations or derived images are redistributed on GitHub or Kaggle. The repository contains code, download instructions and attribution only.
- **Rationale:** A conservative assumption cannot be invalidated by a stricter answer, and not redistributing avoids depending on the answer at all.
- **Consequences:** Open, decided at the relevant stage:
  - (a) the source of the demo GIF/video;
  - (b) how MOT17 reaches Kaggle for fine-tuning (e.g. the kernel downloads from the original URL), and whether private storage on Kaggle counts as redistribution.

## D13 — Own detection evaluator with ignore regions

- **Date:** 2026-10-14
- **Context:** Ultralytics `val()` cannot apply ignore regions, and its only crowd-aware path uses COCO intersection-over-detection-area semantics, which differ from MOT's IoU rule.
- **Options considered:**
  - Ultralytics `val()` without ignore regions;
  - Ultralytics `save_json` with COCO crowd regions;
  - a small own evaluator.
- **Decision:** Implement an own detection evaluator in the pipeline stage, timeboxed to half a day.
  - **Matching per image:**
    - Detections are sorted by score, descending.
    - Each detection is matched to positive GT first (IoU ≥ threshold, one-to-one).
    - Remaining unmatched detections are compared to ignore regions with IoU ≥ 0.5 (MOT semantics, not COCO IoA). If matched, they are removed entirely (neither TP nor FP).
    - Unmatched positive GT counts as FN.
  - **AP:** COCO-style 101-point interpolation. Report AP50 and AP50:95 (10 IoU thresholds), plus precision and recall at the fixed operating confidence.
  - **Unit tests** on hand-computable synthetic cases:
    - a detection on an ignore region;
    - a detection overlapping both a positive and an ignore region;
    - an image with no GT;
    - two detections on one GT.
  - **Cross-check** against pycocotools on data without ignore regions: results must match. Differences on data with ignore regions are expected and are not by themselves evidence of a bug.
- **Rationale:** The rule matches how MOT evaluation treats distractors, and the cross-check anchors the AP computation to a reference implementation.
- **Consequences:**
  - The pipeline stage grows by about half a day, reducing the schedule buffer.
  - The evaluator reads the evaluation threshold from D10's config key, not from the training labels.
