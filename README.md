# cv-detect-track

Real-time multi-person detection and tracking on a 4 GB GPU, run as a controlled speed–accuracy study.

## Problem

How do detector size, input resolution, fine-tuning and the choice of association method trade tracking accuracy (HOTA, IDF1) against end-to-end latency on an edge-class GPU (RTX 3050 Laptop, 4 GB)?

Phase 1 studies pedestrians on MOT17, with every result stratified by static vs moving camera. The code keeps detectors, trackers and datasets behind interfaces (`cvdt.detect.base.Detector`, `cvdt.track.base.Tracker`, `cvdt.data.base.SequenceDataset`). A later phase can therefore add a small-object or drone dataset, or swap the detector for an exported engine, without changing tracking or evaluation code. Design decisions are recorded in [DECISIONS.md](DECISIONS.md) and current progress in [PROJECT_STATUS.md](PROJECT_STATUS.md).

### Evaluation protocol (summary of D2)

- **Tier 1, zero training.** COCO-pretrained detectors, default tracker parameters and thresholds fixed in advance. Evaluated on all 7 unique MOT17 train sequences and reported per sequence and per camera type ([configs/data/tier1_all7.yaml](configs/data/tier1_all7.yaml)).
- **Tier 2, anything learned from MOT17.** Covers fine-tuning, tracker tuning and threshold selection, using a sequence-level train/val/test split whose test set is 1 static + 1 moving sequence. Tier 2 choices use only train/val sequences. Selected split (D8, [configs/data/tier2.yaml](configs/data/tier2.yaml)): train MOT17-04, -05, -13; val MOT17-09, -11; test MOT17-02, -10.
- **Pre-declared selection rules.** The training-label visibility threshold (0.1 vs 0.25) is chosen on val (D9). The fine-tuning input size is chosen from Tier 1 results excluding the Tier 2 test sequences (D11). The detection-evaluation ground truth is fixed in advance and independent of training choices (D10).
- The literature "half" split may appear later as a secondary number, labelled *leaky, for comparability only*.

## Setup

Requires [uv](https://docs.astral.sh/uv/) and an NVIDIA driver supporting CUDA 13.0 (tested with driver 580).

```bash
uv sync
```

```bash
uv run python scripts/check_env.py
```

`check_env.py` prints the versions in use and runs a TrackEval self-test on synthetic data with known answers.

Pinned stack: Python 3.12, `torch==2.14.1+cu130`, `torchvision==0.29.1+cu130`, `ultralytics==8.4.177`, and TrackEval from upstream commit `12c8791`. The reasons for each pin are in [DECISIONS.md](DECISIONS.md) (D6, D7).

## Data preparation

```bash
uv run python scripts/prepare_data.py all
```

`all` runs these steps in order. Each step is idempotent and can also be run on its own:

| Step | What it does | Output (git-ignored unless noted) |
|---|---|---|
| `download` | Checks free disk space, downloads `MOT17.zip` (5.86 GB, resumable), verifies its SHA256 and extracts the train split | `data/raw/` |
| `dedupe` | Hashes every image of the 3 variants (`-DPM`, `-FRCNN`, `-SDP`), compares `gt.txt` and `seqinfo.ini`, and builds one canonical copy per sequence (hard links) | `data/mot17/train/`, `results/dedupe_report.json` (committed) |
| `stats` | Per-sequence statistics, including a global camera-motion estimate | `results/data_stats.csv` (committed) |
| `convert` | YOLO labels (single class `pedestrian`) plus a per-sequence ignore-region sidecar | `data/yolo/mot17/<seq>/{images,labels,ignore.csv}` |
| `render` | 20 random frames (fixed seed) with colour-coded boxes, plus per-sequence camera montages | `reports/figures/data_check/` |

Archive checksum: `MOT17.zip` SHA256 = `7b486bb2af5cfedbfabc020511ff933a3edbf12182ce4985eec1a210f20f9e06` (downloaded 2026-10-14; the server reports Last-Modified 2020-12-18). The archive's publishers do not publish a checksum, so this one pins the copy used here.

Deduplication result: in all 7 sequences, the three variants are byte-identical (every image, `gt.txt`, and `seqinfo.ini` apart from its `name` field). That leaves 5,316 unique frames with no hash shared between sequences.

### Label filtering rule

Implemented in `cvdt.data.filtering`. Class IDs follow the MOT16 paper (Tab. 6). Every ground-truth box falls into exactly one category:

| Category | Rule | Used as |
|---|---|---|
| Positive | class 1 *pedestrian*, flag 1, visibility ≥ `min_visibility_train` (default 0.25, [configs/data/mot17.yaml](configs/data/mot17.yaml)) | YOLO training label |
| Dropped by visibility | class 1, flag 1, visibility < threshold | not a training label; written to the ignore sidecar (`low_visibility`) |
| Ignore region | classes 2 *person on vehicle*, 7 *static person*, 8 *distractor*, 12 *reflection* (any flag); class 1 with flag 0 | ignore sidecar (`distractor_class`, `zero_flag`) |
| Dropped | classes 3 *car*, 4 *bicycle*, 5 *motorbike*, 6 *non-motorized vehicle*, 9 *occluder*, 10 *occluder on the ground*, 11 *occluder full* | not used |

The training visibility threshold applies to training labels only; its value for fine-tuning is a Tier 2 choice between 0.1 and 0.25 (`labels.min_visibility_train_options`, D9). Ultralytics training has no ignore concept, so boxes below the threshold act as background.

Detection evaluation uses its own, fixed threshold (`eval.detection.ignore_visibility_below: 0.25`, D10). Primary protocol: pedestrians below that visibility and the distractor classes are ignore regions. Secondary protocol: all pedestrians are positives. Tracking evaluation reads the raw `gt.txt` and applies TrackEval's standard MOT17 handling, which uses no visibility filter.

In MOT17 the flag is fully determined by class: every class-1 box has flag 1 and every other box has flag 0, so the `zero_flag` path never fires on MOT17. MOT boxes are full-extent and may extend past the image border; YOLO labels are clipped to the image.

## Dataset statistics

From `results/data_stats.csv`:

- **Ped (flag 1)** is the tracking ground truth.
- **Kept** is the training positives at visibility ≥ 0.25.
- **Motion** is the median global translation between consecutive frames, in pixels per frame.
- **H** columns are kept-box heights in pixels.

| Seq | Camera | Motion px/f | Frames | Resolution | FPS | GT tracks | Ped (flag 1) | Kept | Kept % | Vis < 0.25 % | People/frame mean / max | H < 50 % | H 50–150 % | H > 150 % |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| MOT17-02 | static | 0.08 | 600 | 1920×1080 | 30 | 62 | 18,581 | 9,393 | 10.9 | 49.4 | 31.0 / 36 | 7.5 | 59.0 | 33.5 |
| MOT17-04 | static | 0.22 | 1,050 | 1920×1080 | 30 | 83 | 47,557 | 39,821 | 46.4 | 16.3 | 45.3 / 52 | 0.0 | 13.7 | 86.3 |
| MOT17-05 | moving | 4.20 | 837 | 640×480 | 14 | 133 | 6,917 | 4,767 | 5.6 | 31.1 | 8.3 / 15 | 4.0 | 27.4 | 68.6 |
| MOT17-09 | static | 0.24 | 525 | 1920×1080 | 30 | 26 | 5,325 | 3,745 | 4.4 | 29.7 | 10.1 / 13 | 0.0 | 3.2 | 96.8 |
| MOT17-10 | moving | 4.51 | 654 | 1920×1080 | 30 | 57 | 12,839 | 10,560 | 12.3 | 17.8 | 19.6 / 27 | 6.6 | 60.2 | 33.2 |
| MOT17-11 | moving | 4.42 | 900 | 1920×1080 | 30 | 75 | 9,436 | 7,518 | 8.8 | 20.3 | 10.5 / 19 | 0.0 | 20.3 | 79.7 |
| MOT17-13 | moving | 6.30 | 750 | 1920×1080 | 25 | 110 | 11,642 | 10,024 | 11.7 | 13.9 | 15.5 / 32 | 12.8 | 71.2 | 16.0 |
| **Total** | 3 static / 4 moving | | 5,316 | | | 546 | 112,297 | 85,828 | 100 | | | | | |

The per-sequence frame, track and pedestrian-box counts match the official MOT17 table exactly. The official "training totals" row triple-counts the three detection variants: 15,948 frames and 336,891 boxes, i.e. 3 × 5,316 and 3 × 112,297.

## Licenses & data

Facts below were checked on 2026-10-14 against the sources linked.

**Code.** This repository is licensed under [AGPL-3.0](LICENSE), because it depends on Ultralytics, which is AGPL-3.0.

**B1. MOT17 dataset license.**

- **Not verifiable on the primary source as of 2026-10-14.** The MOTChallenge website is offline: [motchallenge.net](https://motchallenge.net/) returns HTTP 410 and states that the benchmark, evaluation server and accounts are offline. Dataset archives and a static leaderboard archive (state of 16 April 2026) stay available at their old addresses.
- No surviving page (e.g. [MOT17 data page](https://motchallenge.net/data/MOT17/)) states a license, and the archive contains no license file.
- Secondary sources consistently report **CC BY-NC-SA 3.0**. This project treats MOT17 as CC BY-NC-SA 3.0: attribution, non-commercial use only, and adaptations shared under the same license.
- No dataset files are redistributed here. Users download MOT17 from the official address with `scripts/prepare_data.py`.
- Until the license is confirmed, no MOT17 frames, annotations or derived images are redistributed on GitHub or Kaggle (D12).

**Demo GIF / figures from MOT17 frames.** A GIF or figure built from MOT17 frames is an adaptation of the dataset, not part of the AGPL code. Under CC BY-NC-SA 3.0 it must:

- credit the dataset,
- not be used commercially, and
- be marked as licensed under CC BY-NC-SA 3.0, separately from the code license.

Rendered frames are git-ignored. The source of any demo media is an open question (D12); if MOT17-derived media is ever published, it will carry that notice.

**Attribution.** MOT17 uses the MOT16 videos with updated annotations. Cite:

> A. Milan, L. Leal-Taixé, I. Reid, S. Roth, K. Schindler. *MOT16: A Benchmark for Multi-Object Tracking.* arXiv:1603.00831, 2016. https://arxiv.org/abs/1603.00831

**B2. Ultralytics code and YOLO26 weights.**

- **Code.** [ultralytics/ultralytics](https://github.com/ultralytics/ultralytics) is AGPL-3.0, and the installed package header and PyPI metadata say the same.
- **Weights.** The YOLO26 weights (`yolo26{n,s,m,l,x}.pt`) are release assets of [ultralytics/assets](https://github.com/ultralytics/assets/releases/tag/v8.4.0), also AGPL-3.0. The [Ultralytics licensing page](https://www.ultralytics.com/license) states that all Ultralytics YOLO trained models fall under AGPL-3.0 by default, and that the license covers models produced by the training code.
- **Consequence.** Fine-tuned weights are AGPL-3.0 as well. A commercial or closed deployment needs an Ultralytics Enterprise license, or a detector from another source behind the `Detector` interface (D5).

**B3. MOT16/MOT17 annotation convention** ([MOT16 paper](https://arxiv.org/abs/1603.00831)).

- **Occluded people get full-extent (amodal) boxes.** Sec. 2.2: when a person is partially occluded, "the extent is estimated based on other available information". Boxes of people cropped by the image border extend beyond the frame. Sec. 2.5: targets are annotated through occlusions as long as their position can be determined.
  - In the data, boxes reach x = −479 and heights up to 1,620 px in a 1,080 px frame.
  - 12,001 training positives extend past the border and are clipped for YOLO.
- **`gt.txt` columns** (Tab. 5): `frame, id, bb_left, bb_top, width, height, conf, class, visibility`.
  - `conf` in ground truth is a flag: 1 = consider, 0 = ignore.
  - Visibility is a ratio in [0, 1], reduced by occlusion and by cropping at the image border.
  - All values, including box coordinates, are 1-based (top-left pixel is (1, 1)); `cvdt` converts to 0-based internally.
- **Classes** (Tab. 6): 1 pedestrian, 2 person on vehicle, 3 car, 4 bicycle, 5 motorbike, 6 non-motorized vehicle, 7 static person, 8 distractor, 9 occluder, 10 occluder on the ground, 11 occluder full, 12 reflection.
  - The Tab. 6 caption calls the class "the 7th column". Tab. 5 and the example rows show it is the 8th.
- **Evaluation handling of distractors** (Sec. 4.1.1, and TrackEval `datasets/mot_challenge_2d_box.py`):
  - Tracker boxes matched (IoU ≥ 0.5, Hungarian) to classes 2, 7, 8 or 12 are removed.
  - Only flag-1 pedestrians are then scored.
  - No visibility filter is applied.

**B4. Ignore regions in Ultralytics `model.val()`.** Not supported in version 8.4.177:

- YOLO-format labels have no ignore flag, and `DetectionValidator` counts every unmatched prediction as a false positive.
- The COCO-JSON loader skips `iscrowd` annotations entirely.
- The only crowd-aware path is `save_json` with a COCO-format ground truth passed to `faster_coco_eval`. It applies COCO crowd semantics (intersection-over-detection-area), which differ from the MOT rule above.

Detection metrics with ignore regions will therefore use a small dedicated evaluator, specified in D13 and planned for the pipeline stage.

**TrackEval.**

- Upstream [JonathonLuiten/TrackEval](https://github.com/JonathonLuiten/TrackEval) (MIT) has had no commits since 2022-11-29. It uses `np.float`, `np.int` and `np.bool`, which NumPy removed in 1.24, so the MOT17 path (IDF1, preprocessing) fails on current NumPy.
- `cvdt.eval.trackeval_compat` restores exactly those builtin aliases before import; no metric code is modified.
- Numerical cross-check against the maintained PyPI fork `trackeval==1.3.0`, run on MOT17-02 and MOT17-13 with synthetic misses, jitter, ID switches and false positives: identical results (max abs difference 0 over 33 metric values).

## Known limitations (draft)

- **Tier 2 val has no small people.** MOT17-09 and -11 contain no kept boxes under 50 px, so val cannot show effects on small people. Input size is therefore chosen from Tier 1 results instead (D11).
- **Ground truth may miss very small, distant people.** A detector at a higher input size may find people the annotation lacks, which would show up as lower precision. To be verified in the failure analysis.
- **Full-extent vs visible-part boxes.** MOT boxes are full-extent (amodal), while COCO-pretrained detectors box the visible part. This lowers the measured recall for occluded people.
- **MOT17 license unconfirmed from the primary source.** CC BY-NC-SA 3.0 is assumed; see D12.

## Results

TBD.
