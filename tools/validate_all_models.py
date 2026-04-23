#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _ensure_src_on_path(root: Path) -> None:
    src_dir = root / "src"
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))


def _parse_models(raw: str) -> list[str] | None:
    text = str(raw).strip()
    if not text:
        return None
    return [item.strip() for item in text.split(",") if item.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate all registered ForeSight models on a unified promotion_data workflow."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="artifacts/validate_all_models",
        help="Directory to write rows.json, summary.json, and summary.md",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Optional comma-separated subset of model keys for focused validation",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Optional dataset root override (same semantics as FORESIGHT_DATA_DIR)",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Runtime device profile for models that support it",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=50,
        help="Emit progress and update progress.json every N completed models",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    root = _repo_root()
    _ensure_src_on_path(root)

    from foresight.services.model_validation import run_registry_training_validation

    parser = build_parser()
    args = parser.parse_args(argv)

    payload = run_registry_training_validation(
        models=_parse_models(str(args.models)),
        data_dir=str(args.data_dir).strip() or None,
        device=str(args.device),
        output_dir=str(args.output_dir),
        progress_every=int(args.progress_every),
    )
    print(payload["summary"], flush=True)
    return 1 if int(payload["summary"]["failed_models"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
