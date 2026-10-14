# Project status

Updated 2026-10-14. Decisions are referenced by ID; see [DECISIONS.md](DECISIONS.md).

## Done

- **Setup + data stage** (D1–D7): reviewed and accepted.
  - Environment, interfaces and TrackEval (with the alias shim).
  - MOT17 downloaded, deduplicated to 7 sequences, statistics computed, YOLO labels written, check frames reviewed.
  - Tests green.
- **Review decisions recorded** (D8–D13):
  - Tier 2 split selected: `configs/data/tier2.yaml` (D8).
  - Training visibility threshold: a Tier 2 choice between 0.1 and 0.25 (D9).
  - Detection-evaluation ground truth fixed in advance (D10).
  - Input-size selection rule (D11).
  - MOT17 license assumption and no redistribution (D12).
  - Own detection evaluator specified (D13).
- **Config keys:** separate keys for the training and evaluation visibility thresholds (`configs/data/mot17.yaml`).

## In progress

- Nothing.

## Blockers / open questions

- MOT17 license confirmation from the dataset maintainers is pending (D12).

## Next (waiting for approval)

- Inference pipeline and MOT writer.
- Detection evaluator with ignore regions (D13).
