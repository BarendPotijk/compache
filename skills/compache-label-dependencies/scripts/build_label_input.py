#!/usr/bin/env python3
"""Use the installed Compache contract implementation."""
import sys
try:
    from compache.cli import main
except ModuleNotFoundError:
    raise SystemExit("Install Compache in this Python environment first: python3 -m pip install -e .")

raise SystemExit(main(["dependencies", "prepare", *sys.argv[1:]]))
