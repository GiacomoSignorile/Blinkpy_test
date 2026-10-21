#!/bin/bash
# Install the powerwatch collector as a systemd service. Run ONCE per PC:   sudo bash install.sh [hub_url]
#
# What it does (nothing else):
#   1. copies collector.py to /opt/powerwatch/ (owned by root, so the service cannot be altered by a normal user)
#   2. lets the user "$SUDO_USER" read ONLY the CPU energy counters in /sys/class/powercap (mode 0440, group of that user).
#      They are root-only by default because of a side-channel risk on shared machines; on a personal PC this is fine.
#   3. runs the collector as that normal user (no root, no extra capabilities), restarted if it dies, started at boot.
# Undo:  sudo systemctl disable --now powerwatch && sudo rm -r /opt/powerwatch /etc/systemd/system/powerwatch.service /etc/tmpfiles.d/powerwatch.conf
set -e
[ "$(id -u)" = 0 ] || { echo "Run with sudo:  sudo bash $0"; exit 1; }
USER_NAME="${SUDO_USER:-giacomo}"
HUB="${1:-http://100.66.9.116:8765/api/power}"
HERE="$(cd "$(dirname "$0")" && pwd)"
install -d -m 755 /opt/powerwatch
install -m 755 "$HERE/collector.py" /opt/powerwatch/collector.py
GROUP_NAME="$(id -gn "$USER_NAME")"
echo "z /sys/class/powercap/intel-rapl*/energy_uj 0440 root $GROUP_NAME -" > /etc/tmpfiles.d/powerwatch.conf
cat > /etc/systemd/system/powerwatch.service <<UNIT
[Unit]
Description=Powerwatch: report this PC's power draw to Blinkwatch
After=network-online.target

[Service]
User=$USER_NAME
Environment=POWER_URL=$HUB
# the leading + runs this one line as root, so the counters are readable even if they appear late at boot
ExecStartPre=+/usr/bin/systemd-tmpfiles --create /etc/tmpfiles.d/powerwatch.conf
ExecStart=/usr/bin/python3 /opt/powerwatch/collector.py
Restart=always
RestartSec=20
NoNewPrivileges=true
ProtectSystem=strict
ProtectHome=read-only
ReadWritePaths=/home/$USER_NAME/.cache

[Install]
WantedBy=multi-user.target
UNIT
mkdir -p "/home/$USER_NAME/.cache" && chown "$USER_NAME:$GROUP_NAME" "/home/$USER_NAME/.cache"
systemctl daemon-reload
systemctl enable --now powerwatch
sleep 2
systemctl --no-pager --lines=5 status powerwatch || true
echo; echo "Fatto. Log in tempo reale:  journalctl -u powerwatch -f"
