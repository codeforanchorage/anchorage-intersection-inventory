"""Pipeline orchestrator: run individual phases or the full pipeline."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import (
    fetch_signals, fetch_imagery, detect_assets, assess_condition,
    export_geojson, visualize_priority,
)  # noqa: E402


def setup_logging(verbose: bool) -> None:
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def run_phase(phase: int, args: argparse.Namespace) -> int:
    if phase == 1:
        print("=== Phase 1: fetch signals ===")
        return fetch_signals.main()
    if phase == 2:
        print("=== Phase 2: fetch imagery ===")
        argv = []
        if args.dry_run:
            argv.append("--dry-run")
        if args.limit is not None:
            argv += ["--limit", str(args.limit)]
        return fetch_imagery.main(argv)
    if phase == 3:
        print("=== Phase 3: detect assets ===")
        argv = ["--backend", args.backend]
        if args.limit is not None:
            argv += ["--limit", str(args.limit)]
        if args.skip_existing:
            argv.append("--skip-existing")
        if args.with_condition:
            argv.append("--with-condition")
        if args.extract_crops:
            argv.append("--extract-crops")
        if args.confidence is not None:
            argv += ["--confidence", str(args.confidence)]
        if args.concurrency is not None:
            argv += ["--concurrency", str(args.concurrency)]
        return detect_assets.main(argv)
    if phase == 4:
        print("=== Phase 4: assess condition ===")
        argv = []
        if args.limit is not None:
            argv += ["--limit", str(args.limit)]
        return assess_condition.main(argv)
    if phase == 5:
        print("=== Phase 5: export geojson ===")
        return export_geojson.main([])
    if phase == 6:
        print("=== Phase 6: visualize priority findings ===")
        argv = []
        if args.limit is not None:
            argv += ["--limit", str(args.limit)]
        return visualize_priority.main(argv)
    raise ValueError(f"unknown phase {phase}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Anchorage intersection asset inventory pipeline.",
    )
    parser.add_argument("--phase", type=int, choices=[1, 2, 3, 4, 5, 6],
                        help="Run a single phase.")
    parser.add_argument("--all", action="store_true",
                        help="Run phases 1 through 6 sequentially.")
    parser.add_argument("--limit", type=int, default=None,
                        help="Process only the first N intersections.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Phase 2: check GSV coverage without downloading.")
    parser.add_argument("--backend", choices=["sam3", "roboflow", "vision-only"],
                        default="sam3",
                        help="Phase 3 detection backend (default: sam3).")
    parser.add_argument("--vision-only", action="store_true",
                        help="Deprecated alias for --backend vision-only.")
    parser.add_argument("--with-condition", action="store_true",
                        help="Phase 3 sam3/roboflow: also send each crop to Claude for condition.")
    parser.add_argument("--extract-crops", action="store_true",
                        help="Phase 3: write per-asset crops under data/results/crops/.")
    parser.add_argument("--confidence", type=float, default=None,
                        help="Phase 3 default detection confidence threshold.")
    parser.add_argument("--concurrency", type=int, default=None,
                        help="Phase 3 vision-only: concurrent in-flight Claude calls (default 1).")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Phase 3: skip intersections with existing detections.")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable DEBUG logging.")
    args = parser.parse_args()
    setup_logging(args.verbose)

    if args.vision_only:
        args.backend = "vision-only"

    if not args.phase and not args.all:
        parser.error("specify --phase N or --all")

    phases = [1, 2, 3, 4, 5, 6] if args.all else [args.phase]
    for p in phases:
        rc = run_phase(p, args)
        if rc != 0:
            print(f"Phase {p} exited with code {rc} — stopping.")
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
