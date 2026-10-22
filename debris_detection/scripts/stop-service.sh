#!/usr/bin/env bash
set -euo pipefail
sudo systemctl stop riverlens.service
echo 'Riverlens stopped. To start: sudo systemctl start riverlens.service'
