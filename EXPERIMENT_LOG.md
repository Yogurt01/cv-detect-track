# Experiment log

One entry per experiment. Every number quoted in the README links back to an entry here. Hypotheses are recorded before the experiments that test them.

## Entry template

```
### <ID> — <short title>
- Date:
- Commit:
- Config: configs/experiments/<ID>.yaml
- Command:
- Raw output: results/<ID>/...
- Result:
- Notes:
```

## Pre-registered hypotheses

### H1 (pre-registered, 2026-10-14)

Removing ByteTrack's second (low-confidence) association stage will reduce HOTA more on MOT17-04 (dense, static, elevated view) than on MOT17-13 (moving camera on a bus), because frequent occlusion produces more low-confidence detections.

Competing hypothesis: on 04 the drop is mainly in DetA (static camera lets Kalman re-associate after short gaps), while on 13 the drop is mainly in AssA (camera motion makes predictions drift during gaps).

Evaluate by reporting DetA and AssA separately.

Note (2026-10-14): H1 is a Tier 1 experiment (COCO-pretrained detector, default parameters), evaluated on all 7 sequences. It must not be run with a fine-tuned detector.

## Entries

### Smoke tests of the pipeline stage (2026-10-16)

All SMOKE entries are **smoke tests, not reported results**. They check that the chain works (frames → detector → cache → tracker → MOT file → TrackEval, plus the detection evaluator). Common settings, unless an entry says otherwise:

- **Code:** commit `8ebf402` plus the uncommitted pipeline-stage changes.
- **Sequences:** MOT17-09 and MOT17-05 only (no Tier 2 test sequences).
- **Detector:** imgsz 640, FP32, external NMS at IoU 0.7 (D16).
- **Tracker:** library defaults, ReID off. Tracker input conf ≥ 0.1 (D18). Track buffer scaled by FPS (D17). Boxes not clipped (D18).
- **Metrics:** TrackEval standard MOT17 handling, in percentage points.
- **Raw outputs:** local and git-ignored, under `outputs/`.

#### Caches used

| Model | Cache directory | Weights SHA256 |
|---|---|---|
| yolo26n | `outputs/detections/yolo26n_640_fp32_conf0.01_14620be7692a/` | `9b09cc8bf347f0fc8a5f7657480587f25db09b34bf33b0652110fb03a8ad4fef` |
| yolo26s | `outputs/detections/yolo26s_640_fp32_conf0.01_b0d4dab5be5f/` | `646f8bc3fe0a656803d95c294f7852321748cb29d13466a1af8862e2db384a1b` |

Both were built with `uv run python scripts/cache_detections.py --weights weights/<model>.pt --imgsz 640 --seqs MOT17-09 MOT17-05`. No frame reached `max_det` (300). The largest per-frame counts were 75 (tight) and 200 (loose).

#### SMOKE-01 to SMOKE-04 — cached tracking, ByteTrack and TrackTrack (smoke test, not a reported result)

- **Command:** `uv run python scripts/run_track.py --cache <cache> --tracker <tracker> --seqs MOT17-09 MOT17-05 --run-name smoke-0N --eval`
- **Raw output:** `outputs/tracks/smoke-0N/{data/,run.json,trackeval.json}`

| ID | Model | Tracker | Seq | HOTA | DetA | AssA | IDF1 | MOTA | IDSW |
|---|---|---|---|---|---|---|---|---|---|
| SMOKE-01 | yolo26n | bytetrack | 09 | 42.87 | 48.53 | 38.17 | 52.97 | 54.08 | 65 |
| SMOKE-01 | yolo26n | bytetrack | 05 | 44.52 | 42.20 | 47.32 | 60.43 | 46.38 | 89 |
| SMOKE-02 | yolo26n | tracktrack | 09 | 42.94 | 44.17 | 41.86 | 55.82 | 51.49 | 35 |
| SMOKE-02 | yolo26n | tracktrack | 05 | 43.12 | 37.51 | 49.68 | 57.88 | 43.65 | 32 |
| SMOKE-03 | yolo26s | bytetrack | 09 | 47.91 | 54.09 | 42.71 | 58.17 | 60.11 | 54 |
| SMOKE-03 | yolo26s | bytetrack | 05 | 43.14 | 43.06 | 43.55 | 57.81 | 45.58 | 104 |
| SMOKE-04 | yolo26s | tracktrack | 09 | 48.70 | 53.94 | 44.12 | 59.39 | 63.64 | 33 |
| SMOKE-04 | yolo26s | tracktrack | 05 | 45.94 | 41.74 | 50.75 | 61.61 | 48.24 | 44 |

#### SMOKE-05 — D17 diagnostic: no FPS scaling of the track buffer (smoke test, not a reported result)

- **Command:** as SMOKE-03 with `--no-fps-scaling` (MOT17-05 buffer 30 frames instead of 14).
- **Raw output:** `outputs/tracks/smoke-05/`
- **Result:**
  - MOT17-05: HOTA 43.35 (vs 43.14), IDF1 58.40 (vs 57.81), IDSW 115 (vs 104).
  - MOT17-09: identical (30 FPS).

#### SMOKE-06 — D18 diagnostic: track boxes clipped to the image (smoke test, not a reported result)

- **Command:** as SMOKE-03 with `--clip-boxes` (the `model.track()` behaviour).
- **Raw output:** `outputs/tracks/smoke-06/`
- **Result:** HOTA 47.77 (09, vs 47.91) and 42.79 (05, vs 43.14).
- **Notes:** Clipping lowers HOTA slightly; the GT boxes are full-extent.

#### SMOKE-07 — detection evaluator on cached detections (smoke test, not a reported result)

- **Command:** `uv run python scripts/eval_det.py --cache <cache> --seqs MOT17-09 MOT17-05 --protocol {primary,secondary} --out outputs/diagnostics/deteval_<model>_<protocol>.json`
- **Raw output:** `outputs/diagnostics/deteval_*.json`

Pooled over MOT17-09 and -05; P and R at conf ≥ 0.25 and IoU 0.5:

| Model | Protocol | AP50 | AP50:95 | P | R | TP | FP | FN | ignored |
|---|---|---|---|---|---|---|---|---|---|
| yolo26n | primary | 92.96 | 60.30 | 84.75 | 90.24 | 7681 | 1382 | 831 | 1397 |
| yolo26n | secondary | 74.17 | 45.23 | 85.49 | 68.39 | 8372 | 1421 | 3870 | 667 |
| yolo26s | primary | 94.97 | 62.55 | 84.46 | 93.93 | 7995 | 1471 | 517 | 1786 |
| yolo26s | secondary | 76.52 | 47.03 | 85.46 | 72.52 | 8878 | 1510 | 3364 | 864 |

#### SMOKE-08 — live run with stage timings (smoke test, not a reported result)

- **Command:** `uv run python scripts/run_track.py --weights weights/yolo26s.pt --imgsz 640 --tracker bytetrack --seqs MOT17-09 --run-name smoke-08-live --timing`
- **Raw output:** `outputs/tracks/smoke-08-live/run.json`
- **Hardware:** RTX 3050 Laptop, driver 580.178.04. Plugged in; power mode not recorded.
- **GPU state:**
  - Before: P0, SM 1500 MHz, memory 6001 MHz, 48 °C.
  - After: P0, SM 1792 MHz (max 2100), 54 °C.
- **Result:** Medians in ms over 515 frames, after skipping the first 10:

| Stage | Median (ms) | p95 (ms) |
|---|---|---|
| read | 6.80 | 9.48 |
| preprocess | 0.86 | 0.97 |
| inference | 6.03 | 6.30 |
| postprocess | 0.95 | 1.38 |
| detect_total | 8.25 | 8.80 |
| track | 0.85 | 0.98 |
| write | 0.06 | 0.09 |
| frame_total | 16.08 | 19.01 |

  That is 62 FPS end-to-end at 1080p input.
- **Notes:**
  - This is not the formal speed protocol.
  - HOTA on MOT17-09 from this live run (47.91) equals SMOKE-03.

#### SMOKE-09 — amodal ignore-match measurement (smoke test, not a reported result)

- **Command:** `uv run python scripts/measure_ignore_match.py --cache <yolo26s cache> --seqs MOT17-09 MOT17-05 --out outputs/diagnostics/ignore_match_yolo26s_640.json`
- **Definition:** A detection is "on" a low-visibility pedestrian (visibility < 0.25) if it is not matched to a positive at IoU 0.5 and at least half of its area lies inside that pedestrian's box.

| Seq | Conf ≥ | Detections on low-vis peds | IoU < 0.5 with that box | Share | Counted as FP (D13) | Median IoU |
|---|---|---|---|---|---|---|
| 09 | 0.25 | 522 | 149 | 28.5% | 136 | 0.587 |
| 05 | 0.25 | 884 | 354 | 40.0% | 346 | 0.538 |
| 09 | 0.10 | 1016 | 476 | 46.9% | 449 | 0.512 |
| 05 | 0.10 | 1506 | 752 | 49.9% | 742 | 0.500 |

- **Notes:** At conf ≥ 0.25 these are 136 of 480 FPs (28%) on 09 and 346 of 991 (35%) on 05 under the primary protocol (yolo26s, SMOKE-07 per-sequence numbers). D13 is unchanged.

### Smoke tests after D16–D20 (2026-10-16)

Same common settings as above. **Smoke tests, not reported results.**

#### SMOKE-10 — caches rebuilt with the single-forward-pass loose NMS (smoke test, not a reported result)

- **Command:** `uv run python scripts/cache_detections.py --weights weights/<model>.pt --imgsz 640 --seqs MOT17-09 MOT17-05 --overwrite`
- **Result:** All four `.npz` files (yolo26n and yolo26s, 09 and 05) are array-identical to the earlier two-pass caches. Caching took 7–10 s per sequence instead of 10–17 s.
- **Notes:** SMOKE-01 to -09 therefore stand unchanged.

#### SMOKE-11 — low-visibility ignore boxes under D19 vs D13 (smoke test, not a reported result)

- **Command:** `uv run python scripts/measure_ignore_match.py --cache <yolo26s cache> --seqs MOT17-09 MOT17-05 --out outputs/diagnostics/ignore_match_d19_yolo26s_640.json`
- **Definition:** "On low-vis" means unmatched and IoA ≥ 0.5 with a low-visibility pedestrian box; counts at IoU 0.5.

| Rule | Seq | Conf ≥ | All FP | On low-vis | On low-vis counted as FP | Share of all FP | Guard FP | Guard overrode an ignore region |
|---|---|---|---|---|---|---|---|---|
| D13 | 09 | 0.25 | 480 | 522 | 136 | 28.3% | – | – |
| D13 | 05 | 0.25 | 991 | 884 | 346 | 34.9% | – | – |
| D19 | 09 | 0.25 | 383 | 522 | 39 | 10.2% | 149 | 39 |
| D19 | 05 | 0.25 | 658 | 884 | 13 | 2.0% | 78 | 13 |
| D13 | 09 | 0.10 | 1626 | 1016 | 449 | 27.6% | – | – |
| D13 | 05 | 0.10 | 2521 | 1506 | 742 | 29.4% | – | – |
| D19 | 09 | 0.10 | 1272 | 1016 | 95 | 7.5% | 384 | 95 |
| D19 | 05 | 0.10 | 1823 | 1506 | 44 | 2.4% | 287 | 44 |

- **Notes:**
  - The D13 rows reproduce SMOKE-09 exactly.
  - Under D19 every remaining FP on a low-visibility box comes from the duplicate guard.

#### SMOKE-12 — detection AP under D19 and D13 (smoke test, not a reported result)

- **Command:** `uv run python scripts/eval_det.py --cache <cache> --seqs MOT17-09 MOT17-05 --protocol {primary,secondary} --out outputs/diagnostics/deteval_d19_<model>_<protocol>.json`
- **Primary protocol:**

| Model | Seq | Rule | AP50 | AP50:95 | P@0.25 | R@0.25 | FP | Ignored (distractor / low-vis) | Guard FP |
|---|---|---|---|---|---|---|---|---|---|
| yolo26n | 09 | D19 | 95.36 | 62.70 | 86.19 | 93.14 | 559 | 531 / 321 | 184 |
| yolo26n | 09 | D13 | 95.04 | 63.02 | 85.34 | 93.14 | 599 | 531 / 281 | – |
| yolo26n | 05 | D19 | 92.66 | 59.10 | 88.57 | 87.96 | 541 | 136 / 691 | 109 |
| yolo26n | 05 | D13 | 91.71 | 58.81 | 84.26 | 87.96 | 783 | 136 / 449 | – |
| yolo26s | 09 | D19 | 97.96 | 66.67 | 90.52 | 97.62 | 383 | 731 / 482 | 149 |
| yolo26s | 09 | D13 | 97.75 | 66.97 | 88.39 | 97.62 | 480 | 731 / 385 | – |
| yolo26s | 05 | D19 | 93.93 | 60.10 | 86.83 | 91.02 | 658 | 134 / 869 | 78 |
| yolo26s | 05 | D13 | 92.78 | 59.66 | 81.41 | 91.02 | 991 | 134 / 536 | – |

- **Pooled (primary), D19 / D13:**
  - yolo26n: AP50 93.64 / 92.96, AP50:95 60.40 / 60.30.
  - yolo26s: AP50 95.72 / 94.97, AP50:95 62.72 / 62.55.
- **Secondary protocol** (all pedestrians positive): D19 and D13 are identical except for one yolo26n detection on MOT17-05 that the guard turns into a FP (P 85.16 vs 85.18).
- **Notes:**
  - D19 raises AP50 and precision.
  - AP50:95 on MOT17-09 is slightly lower under D19 (yolo26n 62.70 vs 63.02; yolo26s 66.67 vs 66.97). At strict IoU thresholds the duplicate guard counts poorly localised boxes on visible people as FPs, which D13 sometimes excused through an overlapping ignore box.

#### SMOKE-13 — speed benchmark, D20 protocol (smoke test, not a reported result)

- **Command:** `uv run python scripts/bench_speed.py --weights weights/yolo26s.pt --imgsz 640 --tracker bytetrack --name smoke-bench-yolo26s-640-fp32-nms-bytetrack`
- **Raw output:** `outputs/bench/smoke-bench-yolo26s-640-fp32-nms-bytetrack.json`
- **Configuration:** yolo26s, 640, FP32, one-to-many head + NMS, ByteTrack, tracker input conf ≥ 0.1. Batch 1, 50 warm-up frames, all frames timed.
- **Hardware state:**
  - Before and after: on AC (ACAD online); power profile `balanced`; CPU governor `powersave` × 16 with EPP `balance_performance`; GPU P0.
  - SM clock 1500 MHz before, 1965 MHz after (max 2100); memory clock 6001 MHz; 47 → 57 °C.

| Seq | Input | Frames | Perception p50 / p95 / p99 (ms) | End-to-end p50 / p95 / p99 (ms) | Decode p50 (ms) | Throughput (FPS) | Peak GPU mem alloc / reserved (MB) | Real-time (D20) |
|---|---|---|---|---|---|---|---|---|
| MOT17-04 | 1920×1080 → 640×384 | 1050 | 9.41 / 10.53 / 12.40 | 15.17 / 17.23 / 21.08 | 5.77 | 62.4 | 99.5 / 120.0 | yes |
| MOT17-05 | 640×480 → 640×480 | 837 | 10.06 / 10.57 / 10.94 | 10.91 / 11.50 / 11.93 | 0.85 | 87.3 | 105.7 / 140.0 | yes |

- **Ultralytics profile** (p50, ms):
  - MOT17-04: preprocess 0.99, inference 5.94, postprocess 0.83; track 1.18.
  - MOT17-05: preprocess 0.40, inference 7.74, postprocess 0.77; track 0.79.
- **Notes:** MOT17-05 has the larger network input (rectangular letterbox: 4:3 → 640×480, 16:9 → 640×384), so its inference is slower than 1080p MOT17-04. It is cheaper end to end only because decoding a 640×480 JPEG is cheap.
