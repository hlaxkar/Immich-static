#!/usr/bin/env python3
"""
extract.py — Backward-compatibility wrapper for `immich-static extract`.
"""

import sys
from immich_static.cli import build_parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args(["extract"] + sys.argv[1:])
    args.func(args)
