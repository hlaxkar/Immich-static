#!/usr/bin/env python3
"""
detect.py — Backward-compatibility wrapper for `immich-static detect`.
"""

import sys
from immich_static.cli import build_parser

if __name__ == "__main__":
    parser = build_parser()
    # Inject 'detect' command if not present
    args = parser.parse_args(["detect"] + sys.argv[1:])
    args.func(args)
