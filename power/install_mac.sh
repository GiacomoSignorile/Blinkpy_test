#!/bin/bash
# Install the powerwatch collector on a Mac (Apple Silicon). Run ONCE, as yourself - NO sudo needed:  bash install_mac.sh [hub_url]
# Starts at login and restarts if it dies. Undo:
#   launchctl bootout gui/$(id -u)/it.powerwatch && rm ~/Library/LaunchAgents/it.powerwatch.plist && rm -r ~/powerwatch
set -e
HUB="${1:-http://100.66.9.116:8765/api/power}"
HERE="$(cd "$(dirname "$0")" && pwd)"
PY="$(command -v python3 || true)"
[ -n "$PY" ] || { echo "python3 non trovato: installa gli Xcode Command Line Tools (xcode-select --install)"; exit 1; }
mkdir -p ~/powerwatch ~/Library/LaunchAgents
cp "$HERE/collector.py" ~/powerwatch/collector.py
cat > ~/Library/LaunchAgents/it.powerwatch.plist <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>it.powerwatch</string>
  <key>ProgramArguments</key><array><string>$PY</string><string>$HOME/powerwatch/collector.py</string></array>
  <key>EnvironmentVariables</key><dict><key>POWER_URL</key><string>$HUB</string></dict>
  <key>RunAtLoad</key><true/><key>KeepAlive</key><true/>
  <key>StandardOutPath</key><string>$HOME/powerwatch/log.txt</string><key>StandardErrorPath</key><string>$HOME/powerwatch/log.txt</string>
</dict></plist>
PLIST
launchctl bootout "gui/$(id -u)/it.powerwatch" 2>/dev/null || true
launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/it.powerwatch.plist
sleep 3
echo "--- primo test: lettura della potenza del Mac"
ioreg -r -n AppleSmartBattery -a | "$PY" -c "
import sys, plistlib
d = plistlib.loads(sys.stdin.buffer.read())[0]
t = d.get('PowerTelemetryData') or {}
print('SystemLoad:', t.get('SystemLoad'), 'mW | ExternalConnected:', d.get('ExternalConnected'), '| PowerTelemetryData presente:', bool(t))"
echo; echo "Fatto. Log: tail -f ~/powerwatch/log.txt"
