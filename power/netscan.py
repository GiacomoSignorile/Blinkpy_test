#!/usr/bin/env python3
"""Netscan: find the devices on the LAN of this PC and report them to the Blinkwatch web app (for the /power page).

Pings every address of the subnet to wake up the ARP table, reads `ip neigh`, resolves names (reverse DNS, then mDNS
via avahi if installed). It sees WHO is online, not how much they draw: the web app attributes a typical wattage per kind.
Run it every few minutes from a timer. Standard library only.
"""
import ipaddress
import json
import os
import shutil
import socket
import subprocess
import urllib.request
from concurrent.futures import ThreadPoolExecutor

URL = os.environ.get("NETSCAN_URL", "http://127.0.0.1:8765/api/power/netscan")


def run(cmd, timeout=10):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def lan():
    """(interface, network) of the default route, e.g. ('wlo1', 192.168.1.0/24); None when offline."""
    parts = run(["ip", "-4", "route", "show", "default"]).split()
    if "dev" not in parts:
        return None
    dev = parts[parts.index("dev") + 1]
    for line in run(["ip", "-4", "-o", "addr", "show", "dev", dev]).splitlines():
        f = line.split()
        if "inet" in f:
            net = ipaddress.ip_interface(f[f.index("inet") + 1]).network
            # /24 or smaller only: never sweep a big VPN-sized range
            return dev, net if net.prefixlen >= 24 else ipaddress.ip_network(f"{net.network_address}/24", strict=False)
    return None


def ping(ip):
    subprocess.run(["ping", "-c", "1", "-W", "1", str(ip)], capture_output=True)


def name_of(ip):
    try:
        return socket.gethostbyaddr(ip)[0]
    except OSError:
        pass
    if shutil.which("avahi-resolve-address"):
        out = run(["avahi-resolve-address", ip], timeout=3).split()
        if len(out) >= 2:
            return out[1]
    return ""


def main():
    found = lan()
    if not found:
        return
    dev, net = found
    with ThreadPoolExecutor(64) as pool:
        list(pool.map(ping, net.hosts()))
    devices = []
    for line in run(["ip", "neigh", "show", "dev", dev]).splitlines():
        f = line.split()
        if "lladdr" not in f or f[-1] in ("FAILED", "INCOMPLETE") or ipaddress.ip_address(f[0]) not in net:
            continue
        devices.append({"ip": f[0], "mac": f[f.index("lladdr") + 1].lower()})
    with ThreadPoolExecutor(16) as pool:
        for d, n in zip(devices, pool.map(lambda d: name_of(d["ip"]), devices)):
            d["hostname"] = n
    req = urllib.request.Request(URL, json.dumps(devices).encode(), {"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=15).read()
    print(f"reported {len(devices)} devices on {net}")


if __name__ == "__main__":
    main()
