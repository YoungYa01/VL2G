# Qualitative cases for VL2G v1

## Improved cases

### goose_001
- GT type: loose
- Observation: The loose GT includes a large roadside grass / field area as traversable, while VL2G produces a cleaner road-focused prediction.
- Baseline issue: The pseudo-only prediction is over-expanded and includes broad non-road regions near the roadside.
- VL2G improvement: VL2G suppresses the ambiguous grassy shoulder and better preserves the actual drivable road boundary.

### goose_002
- GT type: loose
- Observation: The GT mask covers a wide grassy area beside the road, but VL2G successfully focuses on the visually clear road region.
- Baseline issue: The pseudo-only prediction expands into the left-side grass / open roadside area, producing an overly broad traversable mask.
- VL2G improvement: VL2G reduces this over-expansion and gives a more precise prediction of the road surface, especially near the left boundary.

## Failure cases

### goose_016
- GT type: loose
- Failure: VL2G incorrectly predicts the left-side vehicle as traversable, producing a clear false positive on a non-traversable obstacle.
- Observation: The central cement road is correctly recognized as traversable. The right-side brick-paved region can also be considered traversable under the loose GT setting, since it is a flat paved surface connected to the road.
- Possible reason: The vehicle region has low contrast with the road surface and is adjacent to the traversable area, causing the model to over-smooth the prediction across the object boundary.

### goose_017
- GT type: loose
- Failure: VL2G correctly captures the foreground road surface, but the prediction becomes fragmented in the mid-field grass area and fails to recover the large continuous traversable region annotated in the loose GT.
- Possible reason: The grass field is visually ambiguous and has weak road-like structure, causing the model to produce scattered high-confidence blobs instead of a smooth traversable mask. The loose GT also annotates broad grassy terrain as traversable, which may conflict with pseudo-labels that are biased toward visually salient road surfaces.
# 1000-scale qualitative diagnosis

## Cases to inspect

### Strict-improved cases
- goose_043
- goose_008
- goose_015
- goose_018

### Strict-degraded but loose-improved cases
- goose_027
- goose_052
- goose_053

### Strong failure case
- goose_007

## Analysis template

For each case:
- Scene:
- Strict GT behavior:
- Loose GT behavior:
- Pseudo-only prediction:
- VL2G prediction:
- Score map difference:
- Interpretation:
