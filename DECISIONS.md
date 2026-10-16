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

## D14 — MOT17 license confirmed

- **Date:** 2026-10-16
- **Supersedes:** D12
- **Context:** D12 assumed CC BY-NC-SA 3.0 because motchallenge.net is offline. The owner asked the MOTChallenge maintainers at TUM to confirm the license and three specific uses.
- **Options considered:** keep the D12 assumption; adopt the maintainers' written confirmation.
- **Decision:**
  - MOT17 images and annotations are CC BY-NC-SA 3.0, confirmed by the MOTChallenge maintainers (TUM) by email on 2026-10-16. The reply notes that it is the sender's reading of the license, not legal advice.
  - Provenance caveat from the same email: MOT17-05/06 come from the ETH dataset (Ess et al., ICCV 2007) and MOT17-01/02 from a publicly available video; the original sources' conditions may apply in addition.
  - A demo video/GIF from MOT17 frames is allowed as a non-commercial, attributed derivative under CC BY-NC-SA 3.0. Demo material uses only sequences recorded by the MOTChallenge authors (MOT17-04, -11, -13), never 01/02/05/06.
  - A private, non-shared copy on a cloud platform (e.g. Kaggle) for training is allowed. Publishing it as a Kaggle dataset or redistributing it otherwise is not. The exact Kaggle data path is decided at the fine-tuning stage.
  - Required attribution: cite the MOT16 paper (Milan, Leal-Taixé, Reid, Roth, Schindler, arXiv:1603.00831) and link https://motchallenge.net/data/MOT17/.
- **Rationale:** A written statement from the dataset maintainers is the best available primary source while the website is offline. Restricting demos to sequences the maintainers recorded themselves avoids the extra conditions of third-party sources.
- **Consequences:**
  - D12's open items are resolved: demo source (04, 11, 13) and Kaggle (private copy only).
  - The repository still redistributes no MOT17 data.
  - The correspondence is kept privately; no names or contact details appear in public files.

## D15 — Pipeline architecture: detector interface, tracker adapters, detection cache

- **Date:** 2026-10-16
- **Context:** Tracker comparisons, detection evaluation and ablations must see identical detections, and repeated GPU inference is slow on a 4 GB laptop GPU.
- **Options considered:**
  - run the detector live in every experiment;
  - cache detections once per detector configuration and sequence.
- **Decision:**
  - Accuracy experiments (tracker comparison, detection evaluation, ablations) read cached detections, so every tracker sees byte-identical detections.
  - Cache key: (model weights SHA256, imgsz, precision, conf floor, sequence). Settings: conf floor 0.01, COCO class 0 (person) only, FP32, batch 1, default max_det. Record whether any frame hits max_det. Each consumer applies its own thresholds on top of the cache.
  - Operating confidence for detection precision/recall: 0.25. Verified in ultralytics 8.4.177: it equals ByteTrack's default `track_high_thresh`. The full verified ByteTrack defaults are `track_high_thresh` 0.25, `track_low_thresh` 0.1, `new_track_thresh` 0.25, `track_buffer` 30, `match_thresh` 0.8, `fuse_score` True.
  - Speed benchmarks run live end-to-end, never from cache.
  - Trackers receive the sequence FPS from `seqinfo.ini` (MOT17-05 is 14 FPS), because `track_buffer` is counted in frames (implementation: D17).
  - ReID stays off (the default) for all trackers in Tier 1.
  - A cache-vs-live consistency test must pass before any official experiment.
- **Rationale:** Caching removes detector nondeterminism and cost from tracker comparisons. Speed must include the detector, so it is always measured live.
- **Consequences:**
  - Cache files live under the git-ignored `/outputs/detections/`, with a metadata JSON per cache key.
  - Ultralytics may be imported only in `src/cvdt/detect/` and in the tracker adapter module `src/cvdt/track/ultralytics_trackers.py`.

## D16 — YOLO26 inference mode and cache contents

- **Date:** 2026-10-16
- **Status:** Proposed during implementation; approved by the project owner on 2026-10-16.
- **Context:**
  - YOLO26 checkpoints carry two heads: a one-to-one (NMS-free) head and a one-to-many head.
  - In ultralytics 8.4.177 the predictor argument `nms` defaults to `None`, meaning "external NMS". The predictor then runs the one-to-many head followed by standard NMS (IoU 0.7). The NMS-free head is used only with `nms=False`. This also applies to `model.track()`.
  - TrackTrack's track-aware initialisation (TAI) uses detections that a tight NMS dropped but a loose NMS (IoU 0.95) kept. `model.track()` recovers them from internal predictor state.
- **Options considered:**
  - library default (one-to-many head + NMS);
  - NMS-free one-to-one head (`nms=False`);
  - drop TAI recovery for TrackTrack.
- **Decision:**
  - Use the library default: one-to-many head + NMS at IoU 0.7.
  - The cache also stores a second, loose-NMS pass (IoU 0.95) per frame. The TrackTrack adapter derives TAI's recovered detections from it with the library's own rule: loose boxes whose max IoU with any tight box is < 0.97.
  - The cache key is extended with the NMS IoU, the loose-NMS IoU, the class filter and max_det.
- **Rationale:** Tier 1 uses library defaults, and `model.track()` uses this mode. Keeping TAI recovery makes the library TrackTrack comparison faithful.
- **Consequences:**
  - The loose-NMS pass re-runs only NMS on the raw predictions of the single forward pass, as `model.track()` does, so caching and live tracking cost one forward pass per frame. The first implementation ran a second forward pass and was corrected on 2026-10-16.
  - The NMS-free head (`nms=False`) is added to the speed study as a measured variant. It is not used in Tier 1 accuracy experiments.
  - On MOT17-09 (yolo26s, 640, 50 frames) both modes give similar confidence distributions: about 1.6 person detections per frame in [0.1, 0.25) and 7.6 at ≥ 0.25 with NMS, versus 1.9 and 7.1 NMS-free. ByteTrack's second-stage band is thin.

## D17 — Frame-rate scaling of the track buffer

- **Date:** 2026-10-16
- **Status:** Proposed during implementation; approved by the project owner on 2026-10-16.
- **Context:**
  - D15 requires trackers to receive the sequence FPS.
  - In ultralytics 8.4.177 none of the six trackers accepts a frame rate: the lost-track lifetime is `track_buffer` frames regardless of FPS, and `model.track()` never passes FPS.
  - Original ByteTrack uses `int(frame_rate / 30 * track_buffer)` frames.
- **Options considered:**
  - use `track_buffer` frames unscaled (identical to `model.track()`);
  - scale `track_buffer` by FPS / 30 in the adapter, for all six trackers.
- **Decision:** The adapter sets `track_buffer = int(fps / 30 * track_buffer)` before constructing any tracker, so the default of 30 frames means 1 s at every frame rate. The scaling can be switched off (`scale_buffer_by_fps=False`).
- **Rationale:**
  - It implements D15's intent (a buffer defined in time, not frames) with the original ByteTrack formula. MOT17-05 (14 FPS) would otherwise keep lost tracks for 2.1 s, and MOT17-13 (25 FPS) for 1.2 s.
  - `track_buffer` is meant as roughly one second at 30 FPS. Without scaling, a 14 FPS sequence keeps lost tracks about twice as long in seconds. This is a documented deviation from the library default.
  - Justified by data conventions and tracker semantics, not by smoke-run metrics; SMOKE-05/06 only quantify the effect.
- **Consequences:**
  - Identical to `model.track()` at 30 FPS (02, 04, 09, 10, 11).
  - Differs on MOT17-05 (14 frames) and MOT17-13 (25 frames).

## D18 — Pipeline implementation defaults

- **Date:** 2026-10-16
- **Status:** Proposed during implementation; approved by the project owner on 2026-10-16.
- **Context:** Several thresholds and conventions were not fixed by D10, D13 or D15 but affect the numbers.
- **Options considered:** per item, as listed under Decision.
- **Decision:**
  - **Tracker input threshold 0.1.** Tracker input is the cached detections filtered at conf ≥ 0.1, the default `conf` that `model.track()` uses ("trackers need low-confidence input"). Each tracker then applies its own thresholds.
  - **GT clipped for detection evaluation.** Ground-truth boxes are clipped to the image before IoU in detection evaluation, since detectors cannot predict outside the frame. Tracking evaluation (TrackEval) uses the unclipped GT.
  - **Secondary protocol keeps distractors as ignore regions.** D10's secondary protocol makes all pedestrians positive, but distractor classes (2, 7, 8, 12) remain ignore regions; only the visibility rule changes.
  - **HOTA in percentage points.** HOTA, DetA, AssA, IDF1 and MOTA are reported in percentage points (×100), the unit of the 1.0-point rules in D9 and D11.
  - **Unclipped track boxes.** Track boxes are written as the tracker's own state, which may extend beyond the image, because the MOT17 GT is full-extent. `model.track()` clips them; `clip_boxes=True` reproduces that exactly.
- **Rationale:**
  - 0.1 matches the library's own tracking path, which keeps Tier 1 "default".
  - Clipping avoids penalising correct detections of border-cropped people.
  - The secondary protocol isolates the visibility effect.
  - Justified by data conventions and tracker semantics, not by smoke-run metrics; SMOKE-05/06 only quantify the effect.
- **Consequences:**
  - The cache-vs-live test uses 0.1 on both paths.
  - Detection AP is not comparable to numbers computed on unclipped GT.

## D19 — Ignore matching for low-visibility pedestrians

- **Date:** 2026-10-16
- **Supersedes:** D13, for the ignore-matching rule only; everything else in D13 stands (matching order, AP computation, operating point, unit tests, pycocotools cross-check).
- **Context:**
  - D13 removes an unmatched detection if its IoU with an ignore region is ≥ 0.5.
  - Ignore regions for low-visibility pedestrians (visibility < 0.25, D10) are full-body (amodal) boxes, while detectors box the visible part.
  - With yolo26s at 640 and conf ≥ 0.25, detections on such pedestrians that miss the IoU rule were 28% (MOT17-09) and 35% (MOT17-05) of all false positives (SMOKE-09).
- **Options considered:**
  - keep D13 (IoU ≥ 0.5 for all ignore regions);
  - IoA for all ignore regions (COCO crowd style);
  - IoU for distractor classes and IoA for low-visibility pedestrians, with a duplicate guard.
- **Decision:** Per image and per IoU threshold, after one-to-one matching to positive GT, each unmatched detection is handled in this order:
  1. **Duplicate guard.** An unmatched detection with IoU ≥ 0.5 against any positive GT box is a false positive and is never excused by an ignore region.
  2. Otherwise, IoU ≥ 0.5 with a distractor-class ignore box (classes 2, 7, 8, 12) → removed (MOT precedent).
  3. Otherwise, IoA ≥ 0.5 with a low-visibility pedestrian ignore box (visibility < 0.25, D10) → removed, where IoA = area(det ∩ ignore) / area(det).
  4. Otherwise, false positive.
- **Rationale:**
  - The geometric mismatch was measured: detectors box the visible part, while the ignore box is full-body.
  - Low-visibility ignore regions are this project's own rule (D10), with no MOT precedent, so the IoU precedent applies only to distractors.
  - The duplicate guard closes the loophole that IoA would otherwise open, e.g. a duplicate box on a visible person lying inside the full-body box of an occluded person behind them, as in MOT17-02 frame 364.
  - Under IoU-only matching, a fine-tuned model that learns full-body boxes would gain precision partly from the evaluation rule itself, confounding the pretrained-vs-fine-tuned comparison.
- **Transparency:** The rule changed after smoke AP values were seen. The change is justified by the measured mechanism, not by AP. Every detection result is also reported under the D13 rule as a sensitivity analysis.
- **Consequences:**
  - The evaluator implements both rules (`ignore_rule="d19"`, the default, and `"d13"`).
  - `scripts/eval_det.py` reports both.
  - In the secondary protocol all pedestrians are positives, so only the duplicate guard and the distractor rule apply.

## D20 — Real-time definition and speed protocol

- **Date:** 2026-10-16
- **Context:** D11 and the speed study need a fixed definition of "real-time". Frame decoding (JPEG read) was about 40% of per-frame time in SMOKE-08, so the definition must say whether it is included.
- **Options considered:**
  - end-to-end latency including decoding;
  - perception latency from a decoded frame;
  - throughput FPS;
  - mean vs tail latency.
- **Decision:**
  - **Primary (perception latency).** Perception latency = preprocess + inference + postprocess + tracking, measured per frame from a decoded frame in memory, with CUDA synchronisation. Real-time if p95 ≤ 33.3 ms (30 FPS, the highest source frame rate in MOT17).
  - **Secondary.** End-to-end latency including frame read / JPEG decode, and throughput FPS, are reported but not used for the threshold.
  - **Excluded.** Rendering/visualisation and file writing are excluded.
  - **Protocol:**
    - Batch 1, 50 warm-up frames, ≥ 500 timed frames.
    - Report p50 / p95 / p99 and peak GPU memory.
    - Record AC power status, power profile, CPU governor, GPU clocks and performance state before and after. These are recorded only, system settings are not changed, and the benchmark stops if not on AC power.
  - **Benchmark sequences.** MOT17-04 (1,050 frames, 1080p, densest scene, worst case for p95) and MOT17-05 (640×480).
- **Rationale:**
  - Decoding cost depends on the input source (JPEG files here, a camera or video decoder in deployment), not on the models under study.
  - A p95 deadline reflects frames that arrive late, which a mean hides.
  - 30 FPS is the fastest source in MOT17, so meeting it covers every sequence.
- **Consequences:**
  - D11's real-time condition uses this definition.
  - `scripts/bench_speed.py` implements the protocol.
