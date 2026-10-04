"""Phone notifications through ntfy, personalised by who was recognised.

Rules (see config.py / .env):
  * NOTIFY_CATEGORIES   what to notify about (persona, animale, veicolo)
  * NOTIFY_MUTE         names that never notify, e.g. NOTIFY_MUTE=Mamma,Papà,Luna
  * strangers and unrecognised people are always notified, with the highest priority
  * cooldown per camera and name, and no notification for old clips (backlog, reclassification)

Test:  python -m blinkwatch.notify        (sends a test message to the configured topic)
"""
import json
import logging
import time
import urllib.request
from datetime import datetime, timezone

from . import config

log = logging.getLogger("notify")
_last_sent = {}  # (camera, text) -> monotonic time

# ntfy priorities: 1 min .. 5 max (urgent, with sound even in do-not-disturb if the user allows it)
PRIORITY = {"unknown": 5, "uncertain": 4, "known": 3, "generic": 3}
ANIMAL_PRIORITY = {"unknown": 4, "uncertain": 2, "known": 2, "generic": 2}


def _publish(payload):
    req = urllib.request.Request(config.NTFY_URL, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    urllib.request.urlopen(req, timeout=10).read()


def build_items(categories, identities):
    """One dict per thing worth telling about, after the mute rules.

    key = identity without the percentage (what the cooldown is tracked on).
    """
    items = []
    for ident in identities:
        kind, state, name = ident["kind"], ident["state"], ident["name"]
        if kind not in config.NOTIFY_CATEGORIES or (state in ("known", "generic") and name in config.NOTIFY_MUTE):
            continue
        animal = kind == "animale"
        shown = f"{name} {round(ident['conf'] * 100)}%" if state == "known" else name
        items.append({"key": f"{kind}:{name}", "text": shown, "kind": kind,
                      "priority": (ANIMAL_PRIORITY if animal else PRIORITY)[state],
                      "cooldown": config.NOTIFY_COOLDOWN_ANIMAL if animal else config.NOTIFY_COOLDOWN})
    if "veicolo" in categories and "veicolo" in config.NOTIFY_CATEGORIES:
        items.append({"key": "veicolo", "text": "Veicolo", "kind": "veicolo",
                      "priority": config.NOTIFY_PRIORITY["veicolo"], "cooldown": config.NOTIFY_COOLDOWN})
    return items


def send_for_clip(row, categories, details, identities):
    """Notify for a freshly classified clip, subject to the rules above."""
    if not (config.NTFY_URL and config.NTFY_TOPIC):
        return
    age = (datetime.now(timezone.utc) - datetime.fromisoformat(row["created_at"])).total_seconds()
    if age > config.NOTIFY_MAX_AGE:
        return
    now = time.monotonic()
    camera = row["camera"].strip()
    items = [i for i in build_items(categories, identities)
             if now - _last_sent.get((row["camera"], i["key"]), -1e9) >= i["cooldown"]]
    if not items:
        return
    title = ", ".join(i["text"] for i in items) + f" – {camera}"
    _publish({
        "topic": config.NTFY_TOPIC,
        "title": title,
        "message": " · ".join(f"{k} {round(v * 100)}%" for k, v in details.items())
                   + f"\n{datetime.fromisoformat(row['created_at']).astimezone():%H:%M:%S}",
        "priority": max(i["priority"] for i in items),
        "tags": sorted({config.NOTIFY_TAGS[i["kind"]] for i in items}),
        "attach": f"{config.PUBLIC_URL}/thumb/{row['id']}",
        "click": config.PUBLIC_URL,
    })
    for i in items:
        _last_sent[(row["camera"], i["key"])] = now
    log.info("notification sent: %s", title)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    _publish({"topic": config.NTFY_TOPIC, "title": "Blinkwatch", "message": "Notifica di prova: funziona!",
              "priority": 3, "tags": ["white_check_mark"], "click": config.PUBLIC_URL})
    print("inviata a", config.NTFY_URL, "topic", config.NTFY_TOPIC)
