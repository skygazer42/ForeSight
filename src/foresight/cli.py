from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="foresight",
        description="ForeSight: time-series forecasting models and utilities.",
    )
    p.add_argument("--version", action="store_true", help="Print version and exit.")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.version:
        # Avoid importing package metadata; keep it simple and dependency-free.
        from . import __version__

        print(f"ForeSight {__version__}")
        return 0

    # Default behavior is just the help text.
    parser.print_help()
    return 0

