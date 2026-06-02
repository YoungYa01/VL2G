# hard risk mask status for goose_train_500

## Generation source

- Source labels: goose_train_500/labels
- Output risk masks: goose_train_500/pseudo_labels/risk_hard_from_labelids
- Risk source: GOOSE labelids
- Risk type: hard obstacle / blocking / water / object risk categories

## Generation statistics

- num_images: 500
- num_written: 500
- num_missing_label: 0
- num_empty_risk: 110
- area_mean: 0.0423
- area_median: 0.0100
- area_p75: 0.0535
- area_max: 0.7273

## Current note

Initial overlap analysis without resizing only matched 245 samples and skipped 255 shape-mismatched samples.
A complete resized overlap analysis is required before training risk-aware VL2G.
