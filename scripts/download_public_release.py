#!/usr/bin/env python3
"""Download and hash-verify a published PathoPress table export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pathopress.public_data import download_public_export  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base_url", help="Base URL containing manifest.json")
    parser.add_argument("destination", type=Path)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    release = download_public_export(args.base_url, args.destination, force=args.force)
    print(
        f"verified={release.root} models={len(release.models)} "
        f"evaluations={len(release.evaluations)} scores={len(release.scores)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
