# Risk-aware VL2G pilot on goose_train_500

## Hard risk mask generation

- source: goose_train_500/labels
- output: goose_train_500/pseudo_labels/risk_hard_from_labelids
- num_written: 500
- num_missing_label: 0
- num_empty_risk: 110
- area_mean: 0.0423
- area_median: 0.0100
- area_p75: 0.0535
- area_max: 0.7273

## Overlap with LISA traversable pseudo labels

- num matched: 500
- resized b to a: 255
- skipped shape mismatch: 0
- mean overlap area: 0.00103
- mean overlap over lisa traversable: 0.00586
- mean overlap over hard risk: 0.03615
- mean IoU: 0.00319
- p75 overlap over lisa traversable: 0.00016
- p90 overlap over lisa traversable: 0.00524
- max overlap over lisa traversable: 0.83475

Interpretation:
Hard risk masks have very low average overlap with LISA traversable pseudo labels.
They are suitable for a small-weight risk suppression pilot, but the max-overlap outlier should be inspected.

## First risk training result

Model:
- VL2G boundary + smooth weak + hard risk
- lambda_boundary: 0.05
- lambda_smooth: 0.01
- lambda_risk: 0.02
- boundary_ce_weight: 0.5
- uncertainty_ce_weight: 0.2

Fixed threshold 0.5 final metrics:
- IoU: 0.6817
- Dice: 0.7679
- Precision: 0.7357
- Recall: 0.8920

Preliminary observation:
At fixed threshold 0.5, the hard-risk model underperforms the previous VL2G weak model.
Threshold sweep and risk-score evaluation are required before drawing conclusions.

## Threshold sweep result for hard risk lambda=0.02

### Strict GT

- best threshold: 0.80
- IoU: 0.6881
- Dice: 0.7739
- Precision: 0.7655
- Recall: 0.8607

### Loose GT

- best threshold: 0.10
- IoU: 0.6087
- Dice: 0.7347
- Precision: 0.8653
- Recall: 0.6953

## Preliminary interpretation

Compared with VL2G weak without risk, hard risk lambda=0.02 decreases strict IoU but improves loose IoU.
This suggests that hard risk supervision changes the score-map preference and may help broader loose traversability, but it currently harms strict safe-core segmentation.
Risk score evaluation is needed to determine whether the strict-IoU drop is compensated by better risk suppression.

## Risk score evaluation

| Model | safe | obstacle ↓ | water ↓ | unsafe ↓ | non-trav ↓ |
|---|---:|---:|---:|---:|---:|
| LISA-500 pseudo-only | 0.9165 | 0.0164 | 0.0311 | 0.0686 | 0.0686 |
| VL2G-500 weak | 0.9165 | 0.0166 | 0.0310 | 0.0680 | 0.0680 |
| VL2G-500 hardrisk lambda=0.02 | 0.9112 | 0.0140 | 0.0352 | 0.0727 | 0.0727 |

Interpretation:
Hard risk lambda=0.02 reduces obstacle score, but it also reduces safe score and increases water_risk / unsafe_proxy / non_traversable scores.
Therefore, the current hard risk setting is not suitable as the final risk-aware VL2G configuration.
The next step is to clean the risk mask and reduce lambda-risk.

## Clean hard risk lambda=0.005 result

### Threshold sweep

Strict GT:
- best threshold: 0.85
- IoU: 0.6906
- Dice: 0.7763
- Precision: 0.7679
- Recall: 0.8626

Loose GT:
- best threshold: 0.10
- IoU: 0.6109
- Dice: 0.7389
- Precision: 0.8674
- Recall: 0.6962

### Risk score

- safe_score mean: 0.9187
- obstacle_score mean: 0.0119
- water_risk_score mean: 0.0352
- unsafe_proxy_score mean: 0.0719
- non_traversable_score mean: 0.0719

### Interpretation

Clean hard risk lambda=0.005 improves over the original hard risk lambda=0.02 setting.
It achieves the best loose IoU and the lowest obstacle score, while maintaining the highest safe score.
However, it still underperforms VL2G weak on strict IoU and increases water_risk / unsafe_proxy scores.
This suggests that the current risk supervision is mainly useful for obstacle suppression, but not yet reliable as a unified risk objective.

## Obstacle-only clean risk lambda=0.005 result

### Fixed threshold 0.5

- IoU: 0.6796
- Dice: 0.7677
- Precision: 0.7422
- Recall: 0.8804

### Threshold sweep

Strict GT:
- best threshold: 0.65
- IoU: 0.6807
- Dice: 0.7686
- Precision: 0.7535
- Recall: 0.8653

Loose GT:
- best threshold: 0.10
- IoU: 0.5939
- Dice: 0.7239
- Precision: 0.8628
- Recall: 0.6805

### Risk score

- safe_score mean: 0.8991
- obstacle_score mean: 0.0180
- water_risk_score mean: 0.0294
- unsafe_proxy_score mean: 0.0655
- non_traversable_score mean: 0.0655

### Interpretation

Obstacle-only clean risk lambda=0.005 does not improve obstacle suppression.
It also significantly underperforms VL2G weak on both strict and loose IoU.
Although it lowers water_risk and unsafe_proxy scores, the safe_score also drops strongly.
Therefore, obstacle-only risk supervision is not suitable as the next VL2G variant.

## Final decision for current risk pilot

The current risk-aware experiments suggest that simple risk-loss training is not yet stable enough to replace the main VL2G weak model.

Final comparison:

- VL2G weak remains the main model because it achieves the best strict IoU.
- Clean hard risk lambda=0.005 can be kept as a safety-aware variant because it achieves the best loose IoU and lowest obstacle score.
- Obstacle-only clean risk lambda=0.005 is not retained because it lowers strict and loose IoU and does not improve obstacle score.

Conclusion:
Risk supervision should not be further tuned in the current stage.
Future risk-aware training should use more reliable semantic risk definitions or multi-prompt VLM risk signals instead of direct hard label union.
