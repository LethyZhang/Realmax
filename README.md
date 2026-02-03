# RealMax

RealMax is a deployment-motivated benchmarking toolkit for real-time multimedia inference.

It measures the **Maximum Sustainable Real-time Resolution (MSRR)** of deep models under P95 latency constraints and provides a unified **RealMax Score** aggregated across multiple FPS targets (60/30/24).

Unlike conventional proxy metrics (e.g., FPS at fixed low resolution), RealMax directly characterizes the latency-bounded resolution frontier that governs real-world streaming deployments.

---

## Features

- Deployment-aligned metric: Maximum Sustainable Real-time Resolution (MSRR)
- P95 latency–based feasibility criterion
- Coarse-to-fine resolution search with OOM handling and conservative rollback
- RealMax Score aggregation across 60 / 30 / 24 FPS
- Batch-1, streaming-oriented evaluation
- Optional average-FPS sweep for diagnostic analysis
- Works with arbitrary PyTorch models (CNNs, Transformers, hybrid backbones, detectors)

RealMax targets model-level inference and is intended as a deployment-motivated proxy rather than a full end-to-end system benchmark.

---

## Installation

### Requirements

- Python >= 3.8  
- PyTorch  
- CUDA (optional, recommended)  
- numpy  
- pillow  
- thop  

Install dependencies:

```bash
pip install torch numpy pillow thop


## Quick Usage

Make sure your PyTorch model is on the target device and in eval mode:

import torch
import realmax

model = ...  # torch.nn.Module
model.cuda().eval()
1. Minimal MSRR Evaluation
mp = realmax.msrr(model)
print(f"Result: {mp} MP")
2. Enforce Resolution Alignment (e.g., multiple of 8)
mp = realmax.msrr(model, size_multiple=8)
3. Enable Debug Logging
mp = realmax.msrr(model, debug=True)
4. Compute RealMax Score (60 / 30 / 24 FPS)
mp_score = realmax.score(model, size_multiple=32, debug=True)
print(f"RealMax Score: {mp_score}")
5. Average FPS Sweep (Optional Diagnostics)
mp_ave_fps = realmax.ave_fps(model, size_multiple=32, debug=True)
mp_ave_fps returns a list of dictionaries, each containing:

