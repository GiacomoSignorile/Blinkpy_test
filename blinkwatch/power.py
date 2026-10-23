"""Power draw of the PCs: collects samples from power/collector.py and totals the energy per month.

Measured by the collectors: CPU package (RAPL), GPU (nvidia-smi). The rest of the PC (board, RAM, disks, fans) and
the power-supply losses are an adjustable ESTIMATE per host: wall power = (cpu + gpu + base_w) / eff.
Calibrate base_w / eff once with a plug meter if you get one:  UPDATE hosts SET base_w=..., eff=... WHERE host=...
"""
import sqlite3
import time
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Body
from fastapi import HTTPException
from fastapi.responses import HTMLResponse, PlainTextResponse

from . import config

router = APIRouter()
DB = config.DATA_DIR / "power.db"
SCHEMA = """
CREATE TABLE IF NOT EXISTS samples (
    ts INTEGER, host TEXT, interval_s REAL, cpu_w REAL, gpu_w REAL, batt_w REAL, ac INTEGER, has_batt INTEGER,
    sys_w REAL,  -- whole-machine power when the OS reports it (Apple Silicon Macs); then cpu/gpu/base are not added
    PRIMARY KEY (host, ts)
);
CREATE TABLE IF NOT EXISTS hosts (host TEXT PRIMARY KEY, base_w REAL, eff REAL, note TEXT, alias TEXT);
-- other devices on the LAN, seen by power/netscan.py. They have no meter: watts is a typical value per kind, editable.
CREATE TABLE IF NOT EXISTS netdevs (
    mac TEXT PRIMARY KEY, ip TEXT, hostname TEXT, alias TEXT, kind TEXT, watts REAL, hidden INTEGER DEFAULT 0,
    first_seen INTEGER, last_seen INTEGER
);
"""
# typical average draw (W) of a device kind that cannot be measured; the user can override it per device
# (a "pc" with a collector is measured instead, see net_rows)
KIND_W = {"router": 12, "tv": 60, "phone": 3, "tablet": 4, "pc": 40, "printer": 5, "camera": 4, "speaker": 4,
          "console": 30, "nas": 20, "iot": 2, "other": 5}
NET_ONLINE_S = 900  # a device not seen for this long counts as offline

# guesses for what the collectors cannot see: (base watts, efficiency of the power supply / charger)
DEFAULTS = {"laptop": (6.0, 0.85), "desktop": (30.0, 0.88)}


def connect():
    db = sqlite3.connect(DB, timeout=10)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    if "sys_w" not in {r[1] for r in db.execute("PRAGMA table_info(samples)")}:  # database created before sys_w existed
        db.execute("ALTER TABLE samples ADD COLUMN sys_w REAL")
        db.commit()
    if "alias" not in {r[1] for r in db.execute("PRAGMA table_info(hosts)")}:
        db.execute("ALTER TABLE hosts ADD COLUMN alias TEXT")
        db.commit()
    return db


@router.post("/api/power")
def ingest(samples: list[dict] = Body(...)):
    db = connect()
    try:
        for s in samples:
            db.execute("INSERT OR IGNORE INTO samples (ts, host, interval_s, cpu_w, gpu_w, batt_w, ac, has_batt, sys_w) VALUES (?,?,?,?,?,?,?,?,?)",
                       (s["ts"], s["host"], s["interval_s"], s.get("cpu_w"), s.get("gpu_w"), s.get("batt_w"),
                        s.get("ac"), s.get("has_batt", 0), s.get("sys_w")))
            kind = "laptop" if s.get("has_batt") else "desktop"
            db.execute("INSERT OR IGNORE INTO hosts (host, base_w, eff, note) VALUES (?,?,?,?)", (s["host"], *DEFAULTS[kind], f"default guess ({kind})"))
        db.commit()
    finally:
        db.close()
    return {"ok": True, "stored": len(samples)}


@router.get("/api/power/monthly")
def monthly(month: str = ""):
    """Energy per host for a month (YYYY-MM, default the current one), in kWh."""
    now = datetime.now(timezone.utc)
    month = month or now.strftime("%Y-%m")
    start = datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)
    end = start.replace(year=start.year + (start.month == 12), month=start.month % 12 + 1)
    db = connect()
    try:
        out = []
        for h in db.execute("SELECT * FROM hosts ORDER BY host"):
            rows = db.execute("SELECT * FROM samples WHERE host=? AND ts>=? AND ts<?",
                              (h["host"], int(start.timestamp()), int(end.timestamp()))).fetchall()
            def wall(r):
                """Wall power of one sample: whole-machine reading if the OS gives it, else CPU+GPU+base; None = unusable."""
                if r["sys_w"] is not None:
                    return r["sys_w"] / h["eff"]
                if r["cpu_w"] is not None:  # without the CPU counter the total would be meaningless
                    return ((r["cpu_w"] or 0) + (r["gpu_w"] or 0) + h["base_w"]) / h["eff"]
                return None

            ws = [(wall(r), r["interval_s"]) for r in rows if wall(r) is not None]
            seconds = sum(i for _, i in ws)
            joules = sum(w * i for w, i in ws)
            cpu = sum((r["cpu_w"] or 0) * r["interval_s"] for r in rows if r["cpu_w"] is not None and r["sys_w"] is None)
            gpu = sum((r["gpu_w"] or 0) * r["interval_s"] for r in rows if r["cpu_w"] is not None and r["sys_w"] is None)
            last = db.execute("SELECT * FROM samples WHERE host=? ORDER BY ts DESC LIMIT 1", (h["host"],)).fetchone()
            now_ts = int(datetime.now(timezone.utc).timestamp())
            live = None
            if last:  # the latest reading, whatever the month: what the PC draws right now
                w = wall(last)
                live = {"age_s": now_ts - last["ts"], "cpu_w": last["cpu_w"], "gpu_w": last["gpu_w"], "sys_w": last["sys_w"],
                        "wall_w": round(w, 1) if w is not None else None}
            out.append({"host": h["host"], "live": live, "samples": len(rows), "with_cpu_counter": len(ws),
                        "hours_reporting": round(seconds / 3600, 1),
                        "avg_w": round(joules / seconds, 1) if seconds else None,
                        "kwh": round(joules / 3.6e6, 2), "kwh_cpu": round(cpu / 3.6e6, 2), "kwh_gpu": round(gpu / 3.6e6, 2),
                        "base_w": h["base_w"], "eff": h["eff"], "note": h["note"], "alias": h["alias"],
                        "last_seen": datetime.fromtimestamp(max((r["ts"] for r in rows), default=0), timezone.utc).isoformat()
                        if rows else None})
    finally:
        db.close()
    return {"month": month, "hosts": out, "total_kwh": round(sum(h["kwh"] for h in out), 2)}


# wall power of one sample (joined with its host): whole-machine reading if the OS gives it, else CPU+GPU+base; NULL = unusable
WALL_SQL = ("CASE WHEN s.sys_w IS NOT NULL THEN s.sys_w / h.eff "
            "WHEN s.cpu_w IS NOT NULL THEN (s.cpu_w + COALESCE(s.gpu_w, 0) + h.base_w) / h.eff END")


@router.get("/api/power/series")
def series(since: int, until: int = 0, buckets: int = 120):
    """Average power per host in `buckets` equal time slices between since and until (epoch seconds, default now)."""
    until = until or int(time.time())
    if until <= since:
        raise HTTPException(400, "until must be after since")
    step = max(60, (until - since) // max(1, min(buckets, 500)))
    db = connect()
    try:
        rows = db.execute(f"""SELECT s.host, (s.ts - ?) / ? AS b, AVG({WALL_SQL}) AS wall, AVG(s.cpu_w) AS cpu, AVG(s.gpu_w) AS gpu
                              FROM samples s JOIN hosts h ON h.host = s.host WHERE s.ts >= ? AND s.ts < ?
                              GROUP BY s.host, b ORDER BY b""", (since, step, since, until)).fetchall()
    finally:
        db.close()
    hosts, total = {}, {}
    for r in rows:
        if r["wall"] is None:
            continue
        ts = since + r["b"] * step
        hosts.setdefault(r["host"], []).append({"ts": ts, "wall": round(r["wall"], 1),
                                                "cpu": None if r["cpu"] is None else round(r["cpu"], 1),
                                                "gpu": None if r["gpu"] is None else round(r["gpu"], 1)})
        total[ts] = total.get(ts, 0) + r["wall"]
    return {"since": since, "until": until, "step_s": step, "hosts": hosts,
            "total": [{"ts": t, "wall": round(w, 1)} for t, w in sorted(total.items())]}


def guess_kind(hostname: str, mac: str, ip: str) -> str:
    """Best-effort kind of a LAN device from its name; phones usually show up with a private (random) MAC and no name."""
    n = (hostname or "").lower().split(".")[0]  # first label only: every name here ends in the ISP domain
    rules = [("router", ("router", "gateway", "fritz", "modem", "archer", "h388", "ont-", "hub-")),
             ("tablet", ("ipad", "tablet", "tab-", "pad")),
             ("phone", ("iphone", "android", "galaxy", "pixel", "oneplus", "xiaomi", "redmi", "huawei", "poco", "oppo", "realme")),
             ("tv", ("tv", "bravia", "webos", "roku", "chromecast", "firetv", "appletv", "apple-tv")),
             ("speaker", ("echo", "alexa", "homepod", "sonos", "nest-", "google-home", "speaker")),
             ("printer", ("printer", "epson", "brother", "canon", "laserjet", "deskjet")),
             ("camera", ("camera", "cam-", "blink", "tapo", "ring-", "reolink", "eufy")),
             ("console", ("playstation", "ps4", "ps5", "xbox", "nintendo", "switch")),
             ("nas", ("nas", "synology", "qnap")),
             ("pc", ("macbook", "laptop", "desktop", "vivobook", "imac", "-pc", "mac", "ms-7"))]
    for kind, words in rules:
        if any(w in n for w in words):
            return kind
    if ip.endswith(".1"):
        return "router"
    try:
        private = int(mac.split(":")[0], 16) & 2  # locally administered bit: random MAC of a phone/tablet
    except ValueError:
        private = 0
    return "phone" if private else "other"


def host_key(name: str) -> str:
    return (name or "").lower().split(".")[0]


def net_rows(db, now=None):
    """LAN devices with the watts we attribute to them. PCs with a collector are measured, so they count 0 here."""
    now = now or int(time.time())
    measured = {host_key(r["host"]) for r in db.execute("SELECT host FROM hosts")}
    out = []
    for r in db.execute("SELECT * FROM netdevs ORDER BY ip"):
        kind = r["kind"] or guess_kind(r["hostname"], r["mac"], r["ip"] or "")
        is_measured = host_key(r["hostname"]) in measured and kind == "pc"
        online = now - (r["last_seen"] or 0) <= NET_ONLINE_S
        watts = 0 if is_measured else (r["watts"] if r["watts"] is not None else KIND_W.get(kind, 5))
        out.append({"mac": r["mac"], "ip": r["ip"], "hostname": r["hostname"], "alias": r["alias"], "kind": kind,
                    "watts": watts, "watts_custom": r["watts"] is not None, "measured_elsewhere": is_measured,
                    "online": online, "hidden": bool(r["hidden"]), "first_seen": r["first_seen"], "last_seen": r["last_seen"]})
    return out


@router.post("/api/power/netscan")
def netscan(devices: list[dict] = Body(...)):
    """Called by power/netscan.py with what is on the LAN right now: [{mac, ip, hostname}]."""
    now = int(time.time())
    db = connect()
    try:
        for d in devices:
            mac = (d.get("mac") or "").lower()
            if len(mac) != 17:
                continue
            db.execute("INSERT INTO netdevs (mac, ip, hostname, first_seen, last_seen) VALUES (?,?,?,?,?) "
                       "ON CONFLICT(mac) DO UPDATE SET ip=excluded.ip, last_seen=excluded.last_seen, "
                       "hostname=COALESCE(NULLIF(excluded.hostname, ''), hostname)",
                       (mac, d.get("ip"), d.get("hostname") or "", now, now))
        db.commit()
    finally:
        db.close()
    return {"ok": True, "seen": len(devices)}


@router.post("/api/power/netdev/{mac}")
def edit_netdev(mac: str, body: dict = Body(...)):
    """Rename a LAN device, change its kind / typical watts (watts = null goes back to the default of the kind), hide it."""
    db = connect()
    try:
        if not db.execute("SELECT 1 FROM netdevs WHERE mac=?", (mac.lower(),)).fetchone():
            raise HTTPException(404)
        if "watts" in body and body["watts"] is not None and not 0 <= float(body["watts"]) <= 5000:
            raise HTTPException(400, "watts out of range")
        for key in ("alias", "kind", "watts", "hidden"):
            if key in body:
                db.execute(f"UPDATE netdevs SET {key}=? WHERE mac=?", (body[key], mac.lower()))
        db.commit()
    finally:
        db.close()
    return {"ok": True}


@router.post("/api/power/host/{host}")
def edit_host(host: str, body: dict = Body(...)):
    """Rename a PC and calibrate its estimate of the unmeasured part (base_w) and PSU efficiency (eff)."""
    db = connect()
    try:
        if not db.execute("SELECT 1 FROM hosts WHERE host=?", (host,)).fetchone():
            raise HTTPException(404)
        if "base_w" in body and not 0 <= float(body["base_w"]) <= 1000:
            raise HTTPException(400, "base_w out of range")
        if "eff" in body and not 0.3 <= float(body["eff"]) <= 1:
            raise HTTPException(400, "eff out of range")
        for key in ("alias", "base_w", "eff"):
            if key in body:
                db.execute(f"UPDATE hosts SET {key}=? WHERE host=?", (body[key], host))
        db.commit()
    finally:
        db.close()
    return {"ok": True}


@router.get("/api/power/overview")
def overview(since: int = 0):
    """Everything the dashboard shows at once: live reading and energy since `since` (epoch, e.g. local midnight) per PC,
    plus the LAN devices and the power the home draws right now (PCs measured + other devices estimated)."""
    now = int(time.time())
    since = since or now - 86400
    db = connect()
    try:
        energy = {r["host"]: r for r in db.execute(
            f"""SELECT s.host, SUM(({WALL_SQL}) * s.interval_s) AS joules, SUM(s.cpu_w * s.interval_s) AS cpu_j,
                       SUM(s.gpu_w * s.interval_s) AS gpu_j, MAX({WALL_SQL}) AS peak
                FROM samples s JOIN hosts h ON h.host = s.host WHERE s.ts >= ? GROUP BY s.host""", (since,))}
        floor = {r["host"]: r["lo"] for r in db.execute(
            f"""SELECT s.host, MIN({WALL_SQL}) AS lo FROM samples s JOIN hosts h ON h.host = s.host
                WHERE s.ts >= ? AND (s.cpu_w IS NOT NULL OR s.sys_w IS NOT NULL) GROUP BY s.host""", (now - 86400,))}
        hosts = []
        for h in db.execute("SELECT * FROM hosts ORDER BY host"):
            last = db.execute("SELECT * FROM samples WHERE host=? ORDER BY ts DESC LIMIT 1", (h["host"],)).fetchone()
            e = energy.get(h["host"])
            wall = None
            if last:
                wall = (last["sys_w"] / h["eff"]) if last["sys_w"] is not None else (
                    ((last["cpu_w"] + (last["gpu_w"] or 0) + h["base_w"]) / h["eff"]) if last["cpu_w"] is not None else None)
            hosts.append({"host": h["host"], "alias": h["alias"], "base_w": h["base_w"], "eff": h["eff"], "note": h["note"],
                          "has_batt": bool(last["has_batt"]) if last else False, "ac": last["ac"] if last else None,
                          "age_s": now - last["ts"] if last else None, "wall_w": None if wall is None else round(wall, 1),
                          "cpu_w": last["cpu_w"] if last else None, "gpu_w": last["gpu_w"] if last else None,
                          "sys_w": last["sys_w"] if last else None,
                          "kwh": round((e["joules"] or 0) / 3.6e6, 3) if e else 0,
                          "kwh_cpu": round((e["cpu_j"] or 0) / 3.6e6, 3) if e else 0,
                          "kwh_gpu": round((e["gpu_j"] or 0) / 3.6e6, 3) if e else 0,
                          "peak_w": round(e["peak"], 1) if e and e["peak"] is not None else None,
                          "floor_w": round(floor[h["host"]], 1) if floor.get(h["host"]) is not None else None})
        net = net_rows(db, now)
    finally:
        db.close()
    return {"now": now, "since": since, "hosts": hosts, "net": net, "kind_w": KIND_W}


@router.get("/power")
def page():
    return HTMLResponse(Path(__file__).with_name("power.html").read_text(), headers={"Cache-Control": "no-store"})


POWER_FILES = Path(__file__).resolve().parent.parent / "power"


@router.get("/power/{name}", response_class=PlainTextResponse)
def installer(name: str):
    """Serve the collector and its installers, so a PC can fetch them from here (read them before running!)."""
    if name not in ("collector.py", "install.sh", "install_mac.sh", "netscan.py"):
        raise HTTPException(404)
    return (POWER_FILES / name).read_text()
