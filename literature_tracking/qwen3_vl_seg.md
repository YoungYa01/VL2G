# Qwen3-VL-Seg Tracking

Paper:
Qwen3-VL-Seg: Unlocking Open-World Referring Segmentation with Vision-Language Grounding

Status:
- arXiv released: 2026-05-08
- Code: not released yet
- Weights: not released yet

Relevance:
- VLM-native open-world referring segmentation
- Box-guided mask decoder
- No external SAM dependency
- Potential teacher candidate for VL2G

Current plan:
1. Add as latest related work.
2. Build Qwen-Box + SAM2 proxy baseline.
3. Evaluate on GOOSE 56.
4. Train Student-QwenBoxSAM if proxy masks are usable.
5. Revisit when official code/weights are released.
