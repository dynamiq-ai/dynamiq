#!/usr/bin/env python3
"""Example script for hello-world skill. Run from sandbox via SkillsTool run_script."""

import sys


def main() -> int:
    print("Hello from hello-world skill (sandbox).")
    if len(sys.argv) > 1:
        print(f"Arguments: {' '.join(sys.argv[1:])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
