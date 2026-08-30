#!/usr/bin/env python3
"""
immich_sync.py — Backward-compatibility wrapper for `immich-static sync/stats/pull/restore`.
"""

import sys
from immich_static.cli import build_parser

if __name__ == "__main__":
    raw_args = sys.argv[1:]
    parser = build_parser()

    if "--stats" in raw_args or "--storage" in raw_args:
        filtered = [a for a in raw_args if a not in ("--stats", "--storage")]
        args = parser.parse_args(["stats"] + filtered)
    elif "--pull-albums" in raw_args or "--pull" in raw_args:
        filtered = [a for a in raw_args if a not in ("--pull-albums", "--pull")]
        args = parser.parse_args(["pull"] + filtered)
    elif "--restore" in raw_args or "--recalculate" in raw_args:
        filtered = [a for a in raw_args if a not in ("--restore", "--recalculate")]
        args = parser.parse_args(["restore"] + filtered)
    else:
        args = parser.parse_args(["sync"] + raw_args)

    args.func(args)
