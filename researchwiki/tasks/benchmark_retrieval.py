"""Compare BGE and Harrier retrieval quality and local encoding performance."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..benchmark.retrieval_comparison import compare_retrieval, format_comparison
from ..benchmark.retrieval_embeddings import configure_model
from ..fsatomic import write_json_atomic


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(prog="researchwiki benchmark-retrieval", description=__doc__)
    parser.add_argument("--baseline-embedding", default="bge-small")
    parser.add_argument("--candidate-embedding", default="harrier-270m",
                        help="Model ID or alias: harrier-270m, harrier-0.6b, bge-small")
    parser.add_argument("--fixtures", nargs="+", help="Retrieval fixture stems; default: all available")
    parser.add_argument("--backends", nargs="+", choices=("bm25", "semantic", "hybrid"),
                        default=["bm25", "semantic", "hybrid"])
    parser.add_argument("--device", default="cpu", help="Explicit inference device (default: cpu)")
    parser.add_argument("--dtype", choices=("float32", "bfloat16", "auto"), default="float32",
                        help="Inference weight dtype (default: float32, including on CPU)")
    parser.add_argument("--batch-size", type=int, help="Encoding batch size; default: BGE 32, Harrier 8")
    parser.add_argument("--max-seq-length", type=int, help="Cap both models' native context limits")
    parser.add_argument("--cache-folder", help="Optional Hugging Face model cache location")
    parser.add_argument("--timing-repeats", type=int, default=3)
    parser.add_argument("--out", type=Path, help="Write the full JSON comparison to this path")
    parser.add_argument("--json", action="store_true")
    for side in ("baseline", "candidate"):
        parser.add_argument(f"--{side}-trust-remote-code", action="store_true")
        parser.add_argument(f"--{side}-doc-prefix", default="")
        parser.add_argument(f"--{side}-query-prefix", default="")
    args = parser.parse_args(argv)
    try:
        for model in (args.baseline_embedding, args.candidate_embedding):
            configure_model(model, device=args.device, dtype=args.dtype, batch_size=args.batch_size,
                            max_seq_length=args.max_seq_length, cache_folder=args.cache_folder)
        options = [{"trust_remote_code": getattr(args, f"{side}_trust_remote_code"),
                    "doc_prefix": getattr(args, f"{side}_doc_prefix"),
                    "query_prefix": getattr(args, f"{side}_query_prefix")}
                   for side in ("baseline", "candidate")]
        report = compare_retrieval(args.baseline_embedding, args.candidate_embedding,
                                   stems=args.fixtures, backends=tuple(args.backends),
                                   baseline_options=options[0], candidate_options=options[1],
                                   timing_repeats=args.timing_repeats)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    if args.out is not None:
        write_json_atomic(args.out, report)
    print(json.dumps(report, indent=2) if args.json else format_comparison(report))
    return 0
