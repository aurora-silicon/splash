#!/usr/bin/env python3
"""Check the recovered Vulkan source bundle without needing a Vulkan device."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PORT = ROOT / "runtime" / "vulkan"


def main() -> None:
    manifest = json.loads((PORT / "manifest.json").read_text())
    checked = 0
    for entry in manifest["files"]:
        path = ROOT / entry["path"]
        if not path.is_file():
            raise SystemExit(f"missing Vulkan port file: {entry['path']}")
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry["sha256"] or len(data) != entry["bytes"]:
            raise SystemExit(f"stale Vulkan port manifest entry: {entry['path']}")
        checked += 1
    print(f"Vulkan port manifest: {checked} files verified")


if __name__ == "__main__":
    main()
