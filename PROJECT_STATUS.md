# Project status

Updated 2026-10-16. Decisions are referenced by ID; see [DECISIONS.md](DECISIONS.md).

## Done

- **Setup + data stage** (D1–D13): committed.
- **D14–D20 recorded:**
  - D14: MOT17 license confirmed.
  - D15: pipeline architecture.
  - D16–D18: inference mode, FPS-scaled track buffer and pipeline defaults (approved).
  - D19: ignore matching for low-visibility pedestrians.
  - D20: real-time definition and speed protocol.
- **Pipeline stage** (D15):
  - detector wrapper;
  - detection cache;
  - adapters for all six trackers;
  - run loop and MOT writer;
  - TrackEval wrapper;
  - detection evaluator;
  - stage timers.
- **Verification:**
  - Cache vs live is identical, and the adapter reproduces Ultralytics' own tracking for ByteTrack and TrackTrack.
  - Live TrackTrack now runs one forward pass per frame (D16).
- **Detection evaluator** implements D19, with the D13 rule kept for the sensitivity analysis. It matches pycocotools exactly on data without ignore regions.
- **Speed benchmark harness** (D20): `scripts/bench_speed.py`.
- **Smoke runs** SMOKE-01 to SMOKE-13 logged in [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md). They are not results.

## In progress

- Nothing; stopped at the end of this task.

## Blockers / open questions

- None blocking.

## Next (awaiting owner approval)

1. Formal Tier 1 baselines: E0–E2 on all 7 sequences, then the E3 tracker comparison.
2. The H1 ablation (stage-2 association, own ByteTrack).
