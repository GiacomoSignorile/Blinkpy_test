"""Power draw of the PCs: collects samples from power/collector.py and totals the energy per month.

Measured by the collectors: CPU package (RAPL), GPU (nvidia-smi). The rest of the PC (board, RAM, disks, fans) and
the power-supply losses are an adjustable ESTIMATE per host: wall power = (cpu + gpu + base_w) / eff.
Calibrate base_w / eff once with a plug meter if you get one:  UPDATE hosts SET base_w=..., eff=... WHERE host=...
"""
import sqlite3
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
CREATE TABLE IF NOT EXISTS hosts (host TEXT PRIMARY KEY, base_w REAL, eff REAL, note TEXT);
"""
# guesses for what the collectors cannot see: (base watts, efficiency of the power supply / charger)
DEFAULTS = {"laptop": (6.0, 0.85), "desktop": (30.0, 0.88)}


def connect():
    db = sqlite3.connect(DB, timeout=10)
    db.row_factory = sqlite3.Row
    db.executescript(SCHEMA)
    if "sys_w" not in {r[1] for r in db.execute("PRAGMA table_info(samples)")}:  # database created before sys_w existed
        db.execute("ALTER TABLE samples ADD COLUMN sys_w REAL")
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
            db.execute("INSERT OR IGNORE INTO hosts VALUES (?,?,?,?)", (s["host"], *DEFAULTS[kind], f"default guess ({kind})"))
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
                        "base_w": h["base_w"], "eff": h["eff"], "note": h["note"],
                        "last_seen": datetime.fromtimestamp(max((r["ts"] for r in rows), default=0), timezone.utc).isoformat()
                        if rows else None})
    finally:
        db.close()
    return {"month": month, "hosts": out, "total_kwh": round(sum(h["kwh"] for h in out), 2)}


@router.get("/power", response_class=HTMLResponse)
def page():
    return Path(__file__).with_name("power.html").read_text()


POWER_FILES = Path(__file__).resolve().parent.parent / "power"


@router.get("/power/{name}", response_class=PlainTextResponse)
def installer(name: str):
    """Serve the collector and its installers, so a PC can fetch them from here (read them before running!)."""
    if name not in ("collector.py", "install.sh", "install_mac.sh"):
        raise HTTPException(404)
    return (POWER_FILES / name).read_text()
