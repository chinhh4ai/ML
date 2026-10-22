#!/usr/bin/env bash
set -euo pipefail
curl --fail --silent --show-error http://127.0.0.1:5076/api/health
echo
if command -v systemctl >/dev/null && [[ -d /run/systemd/system ]]; then
  systemctl status riverlens.service --no-pager || true
fi
