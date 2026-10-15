"""Generate the fixed comparison set once per prompt strategy (issue #15).

Run from the repo root (GPU pod, or a Mac with --device mps / cpu):
    python experiments/prompt_compare/run_compare.py --strategies simple
    python experiments/prompt_compare/run_compare.py --device mps --duration 5 --cases c02_acoustic_en c03_acoustic_ko
Outputs go to experiments/prompt_compare/outputs/<run_id>/ (wav per case/strategy + results.csv).
"""
import argparse
import csv
import importlib.util
import json
import os
import sys
import time
import types
from datetime import datetime

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, ROOT_DIR)

from dotenv import load_dotenv

from prompt_builder import STRATEGIES, build_description

HERE = os.path.dirname(os.path.abspath(__file__))
MELODY_SAMPLE_RATE = 16000


def install_xformers_stub():
    # xformers is CUDA-only. audiocraft's default 'torch' attention backend computes with torch SDPA
    # and only imports xformers for a compat check and a causal-mask marker, so a stub is enough on mps/cpu.
    if importlib.util.find_spec("xformers") is not None:
        return

    class LowerTriangularMask:
        pass

    xformers = types.ModuleType("xformers")
    ops = types.ModuleType("xformers.ops")
    ops.LowerTriangularMask = LowerTriangularMask
    ops.memory_efficient_attention = None
    xformers.ops = ops
    sys.modules["xformers"] = xformers
    sys.modules["xformers.ops"] = ops


def parse_args():
    parser = argparse.ArgumentParser(description="Compare MusicGen prompt strategies on a fixed input set")
    parser.add_argument("--inputs", default=os.path.join(HERE, "inputs.json"))
    parser.add_argument("--strategies", nargs="+", default=list(STRATEGIES), help=f"available: {', '.join(STRATEGIES)}")
    parser.add_argument("--cases", nargs="+", help="case ids to run (default: all)")
    parser.add_argument("--out-dir", default=os.path.join(HERE, "outputs"))
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "cpu"], help="auto: cuda > mps > cpu")
    parser.add_argument("--duration", type=float, help="override duration_seconds from inputs (shorter is faster on a laptop)")
    parser.add_argument("--no-cache", action="store_true", help="always call the LLM (to measure real latency)")
    parser.add_argument("--dry-run", action="store_true", help="only print the text conditions, skip generation")
    return parser.parse_args()


def main():
    args = parse_args()
    # GEMINI_API_KEY for the LLM strategies
    load_dotenv(os.path.join(ROOT_DIR, ".env"))
    if not args.no_cache:
        # Reuse earlier LLM answers so re-runs don't spend free-tier requests
        os.makedirs(args.out_dir, exist_ok=True)
        os.environ.setdefault("GEMINI_CACHE_PATH", os.path.join(args.out_dir, "llm_cache.json"))
    with open(args.inputs, encoding="utf-8") as f:
        inputs = json.load(f)

    unknown = [s for s in args.strategies if s not in STRATEGIES]
    if unknown:
        sys.exit(f"Unknown strategies: {unknown}. Available: {list(STRATEGIES)}")

    cases = inputs["cases"]
    if args.cases:
        cases = [c for c in cases if c["id"] in args.cases]

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = os.path.join(args.out_dir, run_id)

    model = None
    melody_wav = None
    if not args.dry_run:
        # Heavy imports stay here so --dry-run works without torch/audiocraft
        # app sets the HF/torch cache paths on import and only loads the model on startup
        os.environ.setdefault("TESTING", "true")
        import torch
        import torchaudio
        from audiocraft.models import MusicGen
        from app import MelodyVector, convert_vectors_to_wav_tensor, OUTPUT_SAMPLE_RATE

        os.makedirs(run_dir, exist_ok=True)
        device = args.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        if device != "cuda":
            install_xformers_stub()
        print(f"Loading MusicGen Melody on {device}...")
        model = MusicGen.get_pretrained("facebook/musicgen-melody", device=device)
        if args.duration:
            inputs["duration_seconds"] = args.duration
        model.set_generation_params(duration=inputs["duration_seconds"])
        vectors = [MelodyVector(**v) for v in inputs["melody_vectors"]]
        melody_wav = convert_vectors_to_wav_tensor(vectors, sample_rate=MELODY_SAMPLE_RATE).to(device)
        # Keep the synthesized melody for reference when listening
        torchaudio.save(os.path.join(run_dir, "melody.wav"), melody_wav[0].cpu(), MELODY_SAMPLE_RATE)

    rows = []
    for case in cases:
        for strategy in args.strategies:
            start = time.perf_counter()
            description = build_description(case["genre"], case["mood"], case["prompt"], strategy=strategy)
            build_ms = (time.perf_counter() - start) * 1000
            print(f"[{case['id']}][{strategy}] ({build_ms:.0f}ms) {description}")

            row = {
                "case_id": case["id"],
                "strategy": strategy,
                "genre": case["genre"],
                "mood": case["mood"],
                "prompt": case["prompt"] or "",
                "description": description,
                "build_ms": round(build_ms, 1),
                "generate_ms": "",
                "wav": "",
            }

            if not args.dry_run:
                # Same seed per case so strategies differ only by the text condition
                torch.manual_seed(inputs["seed"])
                start = time.perf_counter()
                outputs = model.generate_with_chroma([description], melody_wav, MELODY_SAMPLE_RATE)
                row["generate_ms"] = round((time.perf_counter() - start) * 1000, 1)
                wav_name = f"{case['id']}__{strategy}.wav"
                torchaudio.save(os.path.join(run_dir, wav_name), outputs[0].cpu(), OUTPUT_SAMPLE_RATE)
                row["wav"] = wav_name

            rows.append(row)

    if args.dry_run:
        return

    with open(os.path.join(run_dir, "results.csv"), "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as f:
        json.dump({"inputs": inputs, "strategies": args.strategies, "cases": [c["id"] for c in cases]}, f, ensure_ascii=False, indent=2)
    print(f"Done. Results in {run_dir}")


if __name__ == "__main__":
    main()
