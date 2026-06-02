# Current VL2G experiment summary

## Stage 1: 500-scale main ablation

| Method | Strict IoU | Loose IoU | Note |
|---|---:|---:|---|
| LISA-500 pseudo-only | 0.6916 | 0.5979 | Baseline |
| Smooth only weak | 0.6917 | 0.5990 | Small gain |
| Boundary only weak | 0.6923 | 0.5981 | Boundary helps more |
| VL2G boundary + smooth weak | 0.6981 | 0.6034 | Main method |

Conclusion:
Boundary + smooth weak is the best main VL2G setting at 500 scale.

## Stage 2: 1000-scale scaling validation

| Train size | Method | Strict IoU | Loose IoU |
|---:|---|---:|---:|
| 500 | LISA pseudo-only | 0.6916 | 0.5979 |
| 500 | VL2G weak | 0.6981 | 0.6034 |
| 1000 | LISA pseudo-only | 0.6788 | 0.6192 |
| 1000 | VL2G weak | 0.6791 | 0.6204 |

Conclusion:
VL2G keeps a small advantage over pseudo-only at both 500 and 1000 scales.
The 1000 subset improves loose IoU but decreases strict IoU, likely due to distribution differences and pseudo-label quality changes.

## Stage 3: Risk-aware pilot

| Method | Strict IoU | Loose IoU | Safe ↑ | Obstacle ↓ | Water ↓ | Unsafe ↓ |
|---|---:|---:|---:|---:|---:|---:|
| VL2G weak | 0.6981 | 0.6034 | 0.9165 | 0.0166 | 0.0310 | 0.0680 |
| Hard risk lambda=0.02 | 0.6881 | 0.6087 | 0.9112 | 0.0140 | 0.0352 | 0.0727 |
| Clean hard risk lambda=0.005 | 0.6906 | 0.6109 | 0.9187 | 0.0119 | 0.0352 | 0.0719 |
| Obstacle-only clean risk lambda=0.005 | 0.6807 | 0.5939 | 0.8991 | 0.0180 | 0.0294 | 0.0655 |

Conclusion:
The main method remains VL2G boundary + smooth weak.
Clean hard risk lambda=0.005 can be reported as a safety-aware variant, but risk supervision is not yet stable enough to become the main method.

## Current final decision

Main model:
VL2G boundary + smooth weak.

Safety-aware variant:
VL2G clean hard risk lambda=0.005.

Stopped line:
Obstacle-only clean risk lambda=0.005 and further risk-loss tuning.

Next stage:
Move from mask-level geometry distillation to multi-prompt semantic consistency and token/embedding-level VL2G v2.
