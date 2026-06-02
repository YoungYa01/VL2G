# goose_train_1000 diagnosis

## Mask area statistics

### goose_train_500 LISA pseudo labels
- num: 500
- empty: 0
- mean area ratio: 0.2472
- std: 0.0965
- min: 0.0128
- p25: 0.1666
- median: 0.2504
- p75: 0.3096
- max: 0.5151

### goose_train_1000 LISA pseudo labels
- num: 1000
- empty: 0
- mean area ratio: 0.2214
- std: 0.0911
- min: 0.0003
- p25: 0.1493
- median: 0.2213
- p75: 0.2826
- max: 0.5409

## Observation

The 1000-scale LISA pseudo labels are smaller on average than the 500-scale pseudo labels.
Therefore, the strict-IoU drop from 500 to 1000 cannot be explained by larger pseudo masks.
A more plausible explanation is that goose_train_1000 is a newly sampled VIS subset rather than a strictly nested extension of goose_train_500, so the training distribution and pseudo-label quality differ.

## Per-image delta observation

Under strict GT, VL2G improves some samples significantly, such as goose_043, goose_008, goose_015, goose_005, and goose_018.
However, it degrades samples such as goose_007, goose_027, goose_006, goose_052, and goose_053.

Under loose GT, some of these cases reverse. For example, goose_043 improves under strict but degrades under loose, while goose_053 and goose_052 degrade under strict but improve under loose.

This suggests that VL2G changes the boundary preference of the traversability score map, rather than uniformly improving all samples.

## Prediction area analysis on goose_sample_56

### Strict threshold setting

- LISA-1000 pseudo-only:
  - threshold: 0.80
  - pred area mean: 0.2270
  - pred area median: 0.2307
  - mean score: 0.2391

- VL2G-1000 weak:
  - threshold: 0.85
  - pred area mean: 0.2182
  - pred area median: 0.2137
  - mean score: 0.2357

Observation:
Under strict evaluation, VL2G predicts a slightly smaller traversable area than pseudo-only.
This is consistent with its higher precision but lower recall in threshold sweep.

### Loose threshold setting

- LISA-1000 pseudo-only:
  - threshold: 0.10
  - pred area mean: 0.2613
  - pred area median: 0.2596
  - mean score: 0.2391

- VL2G-1000 weak:
  - threshold: 0.10
  - pred area mean: 0.2600
  - pred area median: 0.2699
  - mean score: 0.2357

Observation:
Under loose evaluation, the predicted area of VL2G and pseudo-only is almost the same.
Therefore, the slight loose-IoU improvement of VL2G is more likely due to better spatial allocation rather than mask expansion.

Overall interpretation:
VL2G-1000 slightly regularizes the traversability score map and tends to be more conservative under strict thresholds.

## Risk score evaluation on goose_sample_56

### LISA-1000 pseudo-only

- safe_score mean: 0.8944
- obstacle_score mean: 0.0065
- obstacle_gap mean: 0.9005
- water_risk_score mean: 0.0334
- water_risk_gap mean: 0.8708
- unsafe_proxy_score mean: 0.0803
- unsafe_proxy_gap mean: 0.8186
- non_traversable_score mean: 0.0803
- non_traversable_gap mean: 0.8186

### VL2G-1000 weak

- safe_score mean: 0.8933
- obstacle_score mean: 0.0134
- obstacle_gap mean: 0.8843
- water_risk_score mean: 0.0288
- water_risk_gap mean: 0.8744
- unsafe_proxy_score mean: 0.0775
- unsafe_proxy_gap mean: 0.8199
- non_traversable_score mean: 0.0775
- non_traversable_gap mean: 0.8199

## Interpretation

Both models assign high traversability scores to safe regions and very low scores to risk regions.
VL2G weak slightly reduces scores on water_risk, unsafe_proxy, and non_traversable regions, while keeping safe scores almost unchanged.
However, VL2G produces a slightly higher score on obstacle regions than pseudo-only.
Therefore, the current VL2G geometry constraints already provide partial risk suppression, but obstacle-specific risk supervision should be introduced carefully with a small loss weight.

## GT task area statistics on goose_sample_56

### obstacle
- num: 56
- empty: 2
- non_empty: 54
- mean area all: 0.0851
- mean area non-empty: 0.0882
- median non-empty: 0.0434
- p75 non-empty: 0.1137
- max: 0.4083

### water_risk
- num: 56
- empty: 0
- non_empty: 56
- mean area all: 0.3735
- median non-empty: 0.3770
- p75 non-empty: 0.5167
- max: 0.8877

### unsafe_proxy / non_traversable
- num: 56
- empty: 0
- non_empty: 56
- mean area all: 0.6781
- median non-empty: 0.6755
- p75 non-empty: 0.8138
- max: 0.9806

Observation:
unsafe_proxy and non_traversable have identical statistics, so they are likely equivalent masks in the current task setup.
Their area is very large, so they should not be directly used as a strong risk loss target in the first risk-aware training experiment.
