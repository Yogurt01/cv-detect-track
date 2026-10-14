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

None yet.
