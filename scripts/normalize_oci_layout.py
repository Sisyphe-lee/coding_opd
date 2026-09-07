#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from coding_opd.oci_layout import normalize_layout


def main() -> None:
    parser = argparse.ArgumentParser(description="Normalize Docker v2 manifests in an OCI image layout")
    parser.add_argument("layout", type=Path)
    args = parser.parse_args()
    print(json.dumps(normalize_layout(args.layout), sort_keys=True))


if __name__ == "__main__":
    main()
