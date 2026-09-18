#!/usr/bin/env python3
"""
serve.py — Backward-compatibility wrapper for `immich-static serve`.
"""

import sys
from immich_static.cli import build_parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args(["serve"] + sys.argv[1:])
    args.func(args)
