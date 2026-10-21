#!/usr/bin/env bash
# Ubuntu 24.04 x86_64. Install system prerequisites, then project-local dependencies.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ "${1:-}" != '--skip-system' ]]; then
  source /etc/os-release
  if [[ "$ID" != ubuntu || "$VERSION_ID" != 24.04 ]]; then
    echo 'Automatic system installation supports Ubuntu 24.04. Supply Python 3.12, FFmpeg and Node.js >=18.19 yourself, then use --skip-system.' >&2
    exit 1
  fi
  if [[ "$(id -u)" -eq 0 ]]; then elevation=(); else elevation=(sudo); fi
  "${elevation[@]}" apt-get update
  "${elevation[@]}" env DEBIAN_FRONTEND=noninteractive apt-get install -y python3.12 python3.12-venv ffmpeg nodejs npm ca-certificates
fi
python3 -c 'import sys; assert sys.version_info[:2] == (3, 12), "Use Python 3.12 for the pinned dependencies"'
node -e 'const [a,b]=process.versions.node.split(".").map(Number);if(a<18||(a===18&&b<19))throw Error("Node >=18.19 required")'
bash scripts/setup.sh
echo 'Ubuntu setup complete. Edit .env, then: bash scripts/start.sh'
echo 'Optional systemd service: bash scripts/install-service.sh'
