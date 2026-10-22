#!/usr/bin/env bash
# Register a single-process service. Run as the non-root owner of this checkout.
set -euo pipefail
app_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$app_root"
if [[ "$(id -u)" -eq 0 ]]; then
  echo 'Run this script as the regular user who owns the application, with sudo available.' >&2
  exit 1
fi
[[ -x .venv/bin/python && -f frontend/dist/index.html && -f .env ]] || { echo 'Run setup and create .env first.' >&2; exit 1; }
run_user="$(id -un)"
run_group="$(id -gn)"
# Bind loopback by default. Use an SSH tunnel, or deliberately override APP_HOST for a trusted LAN.
app_host="${APP_HOST:-127.0.0.1}"
[[ "$app_host" == 127.0.0.1 || "$app_host" == 0.0.0.0 ]] || { echo 'APP_HOST must be 127.0.0.1 or 0.0.0.0'; exit 1; }
unit_file="$(mktemp)"
trap 'rm -f "$unit_file"' EXIT
.venv/bin/python - "$app_root" "$run_user" "$run_group" "$app_host" > "$unit_file" <<'PY'
import json, sys
root, user, group, host = sys.argv[1:]
if any(c in root for c in '\n\r'):
    raise SystemExit('Application path must not contain line breaks.')
root = root.replace('%', '%%')
q = json.dumps
print(f'''[Unit]
Description=Riverlens CPU aquatic debris analysis
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User={user}
Group={group}
WorkingDirectory={root}
ExecStart={q(root + '/.venv/bin/python')} -m uvicorn backend.main:app --host {host} --port 5076
Environment=PYTHONUNBUFFERED=1
Environment=MPLCONFIGDIR={q(root + '/.data/matplotlib')}
Restart=on-failure
RestartSec=5
TimeoutStopSec=120
KillSignal=SIGINT
UMask=0077
NoNewPrivileges=true
PrivateTmp=true
ProtectSystem=strict
ReadWritePaths={q(root + '/.data')}

[Install]
WantedBy=multi-user.target''')
PY
if [[ "${1:-}" == '--print' ]]; then
  cat "$unit_file"
  exit 0
fi
if [[ -f /etc/systemd/system/riverlens.service ]]; then
  sudo cp -p /etc/systemd/system/riverlens.service "/etc/systemd/system/riverlens.service.backup.$(date +%Y%m%d%H%M%S)"
fi
sudo install -m 0644 "$unit_file" /etc/systemd/system/riverlens.service
sudo systemctl daemon-reload
sudo systemctl enable --now riverlens.service
sudo systemctl status riverlens.service --no-pager
echo 'Check: bash scripts/status.sh'
echo 'Logs: sudo journalctl -u riverlens -f'
