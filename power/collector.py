#!/usr/bin/env python3
"""Powerwatch collector: every minute, measure what this PC draws and send it to the Blinkwatch web app.

Measured: CPU package power (RAPL energy counter), GPU power (nvidia-smi), battery discharge, AC state;
on Apple Silicon Macs the power of the whole machine (sys_w) from the battery controller.
NOT measured: motherboard, RAM, disks, fans, PSU losses - the server adds an adjustable estimate for those.
Samples that cannot be delivered (hub off, network down) are spooled to disk and sent later. Standard library only.
"""
import glob
import json
import os
import plistlib
import platform
import socket
import subprocess
import time
import urllib.request
from pathlib import Path

URL = os.environ.get("POWER_URL", "http://100.66.9.116:8765/api/power")
INTERVAL = int(os.environ.get("POWER_INTERVAL", 60))
MAX_ITER = int(os.environ.get("POWER_MAX_ITER", 0))  # 0 = forever; >0 only for tests
HOST = os.environ.get("POWER_HOST", socket.gethostname())
SPOOL = Path(os.environ.get("POWER_SPOOL", Path.home() / ".cache" / "powerwatch" / "spool.jsonl"))
SPOOL_MAX = 20000  # lines (~2 weeks at one per minute)


def read(path):
    try:
        return Path(path).read_text().strip()
    except OSError:
        return None


def rapl_counter():
    """(energy in microjoules, wrap-around range) of the CPU package, or None if not readable."""
    base = "/sys/class/powercap/intel-rapl:0"
    e, rng = read(f"{base}/energy_uj"), read(f"{base}/max_energy_range_uj")
    return (int(e), int(rng)) if e and rng else None


def gpu_watts():
    try:
        r = subprocess.run(["nvidia-smi", "--query-gpu=power.draw", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=8)
        vals = [float(x) for x in r.stdout.split() if x.replace(".", "", 1).isdigit()]
        return round(sum(vals), 1) if vals else None
    except (OSError, subprocess.SubprocessError):
        return None


def battery():
    """(has_battery, watts drawn from the battery or None unless it is discharging)."""
    bats = glob.glob("/sys/class/power_supply/BAT*")
    if not bats:
        return 0, None
    status, power = read(bats[0] + "/status"), read(bats[0] + "/power_now")
    if status == "Discharging" and power:
        return 1, round(int(power) / 1e6, 1)
    return 1, None


def ac_online():
    for p in glob.glob("/sys/class/power_supply/A*/online"):
        v = read(p)
        if v is not None:
            return int(v)
    return None


def parse_mac_battery(xml: bytes):
    """From `ioreg -r -n AppleSmartBattery -a`: (watts the whole Mac draws, on AC?) on Apple Silicon.

    macOS reports the system load in PowerTelemetryData (milliwatts) without needing root. If that block is missing
    (older macOS) the fallback is the battery discharge, which is only valid when unplugged.
    """
    data = plistlib.loads(xml)
    info = data[0] if isinstance(data, list) else data
    ac = 1 if info.get("ExternalConnected") else 0
    tele = info.get("PowerTelemetryData") or {}
    if tele.get("SystemLoad") is not None:
        return round(tele["SystemLoad"] / 1000, 2), ac
    if not ac and info.get("Voltage") and info.get("InstantAmperage") is not None:
        amps = info["InstantAmperage"]
        amps = amps - 2**64 if amps > 2**63 else amps  # reported as an unsigned 64-bit number
        return round(abs(amps) * info["Voltage"] / 1e6, 2), ac
    return None, ac


def mac_system_watts():
    try:
        out = subprocess.run(["ioreg", "-r", "-n", "AppleSmartBattery", "-a"], capture_output=True, timeout=10).stdout
        return parse_mac_battery(out)
    except Exception:
        return None, None


def send(samples):
    req = urllib.request.Request(URL, data=json.dumps(samples).encode(), headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10).read()


def spooled():
    return [json.loads(l) for l in SPOOL.read_text().splitlines() if l.strip()] if SPOOL.exists() else []


def deliver(sample):
    pending = spooled() + [sample]
    try:
        send(pending)
        SPOOL.unlink(missing_ok=True)
    except Exception as e:  # hub unreachable: keep everything for the next round
        SPOOL.parent.mkdir(parents=True, exist_ok=True)
        SPOOL.write_text("\n".join(json.dumps(s) for s in pending[-SPOOL_MAX:]) + "\n")
        print(f"hub unreachable ({type(e).__name__}); {len(pending)} samples waiting", flush=True)


def main():
    prev, t_prev, n = rapl_counter(), time.monotonic(), 0
    print(f"powerwatch on {HOST} -> {URL} every {INTERVAL}s | CPU counter: {'yes' if prev else 'NOT readable'}", flush=True)
    while True:
        time.sleep(INTERVAL)
        now, t_now = rapl_counter(), time.monotonic()
        cpu_w = None
        if prev and now:
            delta = (now[0] - prev[0]) % prev[1]  # the counter wraps around
            cpu_w = round(delta / 1e6 / (t_now - t_prev), 1)
        has_batt, batt_w = battery()
        sys_w, ac = None, ac_online()
        if platform.system() == "Darwin":  # no RAPL / nvidia-smi / sysfs on a Mac: ask the battery controller
            sys_w, ac = mac_system_watts()
            has_batt = 1
        sample = {"host": HOST, "ts": int(time.time()), "interval_s": round(t_now - t_prev, 1), "cpu_w": cpu_w,
                  "gpu_w": gpu_watts(), "batt_w": batt_w, "ac": ac, "has_batt": has_batt, "sys_w": sys_w}
        prev, t_prev = now, t_now
        deliver(sample)
        n += 1
        if MAX_ITER and n >= MAX_ITER:
            print("samples sent:", sample)
            return


if __name__ == "__main__":
    main()
