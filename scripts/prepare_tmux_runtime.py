#!/usr/bin/env python3
"""Bundle the host tmux and its loader/libs for offline task containers.

Only tmux uses the bundled loader. Task shells and Python keep the image's
original libraries. Mount the directory at /opt/coding-opd-tmux and its tmux
wrapper at /usr/local/bin/tmux, as in uni_agent_react_reference.yaml.
"""

import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    binary = Path(shutil.which("tmux") or "/usr/bin/tmux")
    dependencies = subprocess.check_output(["ldd", str(binary)], text=True)
    libraries = {Path(path) for path in re.findall(r"(/[\w/+.\-]+)", dependencies)}
    loader = next(path for path in libraries if path.name.startswith("ld-linux"))
    destination = args.destination
    destination.mkdir(parents=True, exist_ok=True)
    for source in libraries:
        shutil.copy2(source.resolve(), destination / source.name)
    shutil.copy2(binary.resolve(), destination / "tmux.real")
    wrapper = destination / "tmux"
    wrapper.write_text(
        '#!/bin/sh\nexec /opt/coding-opd-tmux/' + loader.name
        + ' --library-path /opt/coding-opd-tmux /opt/coding-opd-tmux/tmux.real "$@"\n'
    )
    wrapper.chmod(0o755)
    manifest = {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(destination.iterdir()) if path.name != "manifest.json"}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
