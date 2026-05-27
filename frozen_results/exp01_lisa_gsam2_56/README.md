Experiment 01: LISA vs Grounded-SAM2 on 56 GOOSE off-road images

Dataset:
- 56 randomly sampled GOOSE train images with corresponding labelids masks.
- Derived GT: traversable_strict, obstacle, water_risk, unsafe_proxy.

Models:
- Grounded-SAM2: open-vocabulary grounding + SAM2 segmentation baseline.
- LISA: LLM-guided reasoning segmentation teacher candidate.

Main result:
- Grounded-SAM2 / road: IoU 0.139, Recall 0.166
- LISA / traversable: IoU 0.703, Dice 0.775, Recall 0.923

Conclusion:
- LISA is more suitable as a semantic teacher for off-road traversable area pseudo-label generation.
