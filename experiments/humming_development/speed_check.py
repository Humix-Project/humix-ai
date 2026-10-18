"""Measure MusicGen melody load/generation/continuation time on a device: python speed_check.py mps|cpu"""
import sys
import time

import xformers_stub  # noqa: F401
import torch
from audiocraft.models import MusicGen

device = sys.argv[1] if len(sys.argv) > 1 else 'cpu'

start = time.time()
model = MusicGen.get_pretrained('facebook/musicgen-melody', device=device)
print(f"[{device}] load {time.time() - start:.1f}s")

model.set_generation_params(duration=5)
melody = torch.randn(1, 1, 16000 * 5) * 0.1
start = time.time()
out = model.generate_with_chroma(['pop, happy'], melody.to(device), 16000)
print(f"[{device}] generate 5s: {time.time() - start:.1f}s, shape {tuple(out.shape)}, peak {out.abs().max().item():.3f}")

model.set_generation_params(duration=8)
start = time.time()
out2 = model.generate_continuation(out, 32000, ['pop, happy'])
print(f"[{device}] continuation 5s -> 8s: {time.time() - start:.1f}s, shape {tuple(out2.shape)}")
