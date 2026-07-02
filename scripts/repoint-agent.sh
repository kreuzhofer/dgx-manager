#!/usr/bin/env bash
# Repoint a node's dgx-agent at a new manager, via a systemd drop-in (reversible).
#
# Usage:
#   ./repoint-agent.sh <node-ip> <manager-ip>    # point agent at ws://<manager-ip>:4000
#   ./repoint-agent.sh <node-ip> --rollback      # remove drop-in -> unit default (192.168.44.36)
#
# The drop-in overrides Environment=MANAGER_URL in /etc/systemd/system/dgx-agent.service
# without editing the generated unit, so rollback is just removing the file.
set -euo pipefail
NODE="${1:?node-ip}"; TARGET="${2:?manager-ip or --rollback}"
DROPIN=/etc/systemd/system/dgx-agent.service.d/manager-url.conf

if [ "${TARGET}" = "--rollback" ]; then
  ssh -o StrictHostKeyChecking=no daniel@"${NODE}" \
    "sudo rm -f ${DROPIN} && sudo systemctl daemon-reload && sudo systemctl restart dgx-agent"
  echo "rolled back ${NODE} -> unit default"
else
  ssh -o StrictHostKeyChecking=no daniel@"${NODE}" \
    "sudo mkdir -p $(dirname ${DROPIN}) && \
     printf '[Service]\nEnvironment=MANAGER_URL=ws://${TARGET}:4000/ws/agent\n' | sudo tee ${DROPIN} >/dev/null && \
     sudo systemctl daemon-reload && sudo systemctl restart dgx-agent && sleep 2 && \
     systemctl show dgx-agent -p Environment | grep -o 'MANAGER_URL=[^ \"]*'"
  echo "repointed ${NODE} -> ws://${TARGET}:4000"
fi
