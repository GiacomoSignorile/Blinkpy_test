from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from blinkwatch import ipcam_control as c

ROME = ZoneInfo("Europe/Rome")


def at(y, m, d, hh, mm):  # local (Rome) time -> aware datetime; 2026-10-30 is a Friday (weekday 4)
    return datetime(y, m, d, hh, mm, tzinfo=ROME)


def _sched(windows, enabled=True):
    return c.normalize({"schedule": {"enabled": enabled, "windows": windows}})


def test_normalize_clamps_and_ignores_garbage():
    n = c.normalize({"post_s": 9999, "exposure_day": "x", "exposure_night": -50, "sensitivity": "boom", "enabled": "yes",
                     "evil": 1, "alerts": {"offline_min": 0, "battery_pct": 500}})
    assert n["post_s"] == 60 and n["exposure_day"] == c.DEFAULTS["exposure_day"] and n["exposure_night"] == -10
    assert n["sensitivity"] == c.DEFAULTS["sensitivity"] and n["enabled"] is True and "evil" not in n
    assert n["alerts"] == {"offline_min": 1, "battery_pct": 90}


def test_normalize_windows_are_cleaned():
    n = _sched([{"days": [4, 4, 9, "a", True], "start": "25:00", "end": "06:30"}, {"days": []}, "x"])
    assert n["schedule"]["windows"] == [{"days": [4], "start": "22:00", "end": "06:30"}]


def test_disabled_and_pause():
    now = at(2026, 10, 30, 12, 0).astimezone(timezone.utc)
    assert c.is_active(c.normalize({}), now) == (True, "on")
    assert c.is_active(c.normalize({"enabled": False}), now) == (False, "disabled")
    paused = c.normalize({"paused_until": (now + timedelta(hours=1)).isoformat()})
    assert c.is_active(paused, now) == (False, "paused")
    assert c.is_active(paused, now + timedelta(hours=2)) == (True, "on")


def test_daytime_window():
    ctrl = _sched([{"days": [4], "start": "09:00", "end": "17:00"}])  # Fridays 9-17
    assert c.is_active(ctrl, at(2026, 10, 30, 12, 0))[0]
    assert not c.is_active(ctrl, at(2026, 10, 30, 17, 0))[0]  # end is exclusive
    assert not c.is_active(ctrl, at(2026, 10, 31, 12, 0))[0]  # Saturday
    assert c.is_active(ctrl, at(2026, 10, 31, 12, 0)) == (False, "outside_schedule")


def test_window_across_midnight_belongs_to_the_day_it_starts():
    ctrl = _sched([{"days": [4], "start": "22:00", "end": "07:00"}])  # Friday night
    assert c.is_active(ctrl, at(2026, 10, 30, 23, 0))[0]  # Friday 23:00
    assert c.is_active(ctrl, at(2026, 10, 31, 3, 0))[0]  # Saturday 03:00, still Friday's night
    assert not c.is_active(ctrl, at(2026, 10, 31, 8, 0))[0]
    assert not c.is_active(ctrl, at(2026, 10, 30, 3, 0))[0]  # Friday 03:00 is Thursday's night, not listed


def test_schedule_off_means_always():
    ctrl = _sched([{"days": [0], "start": "01:00", "end": "02:00"}], enabled=False)
    assert c.is_active(ctrl, at(2026, 10, 30, 12, 0)) == (True, "on")


def test_dst_change_uses_local_time():
    ctrl = _sched([{"days": list(range(7)), "start": "08:00", "end": "09:00"}])
    assert c.is_active(ctrl, at(2026, 10, 25, 8, 30))[0]  # day the clocks go back
    assert not c.is_active(ctrl, at(2026, 10, 25, 9, 30))[0]


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "CONTROL_FILE", tmp_path / "ctl.json")
    assert c.load() == c.normalize({})  # missing file -> defaults
    c.update({"post_s": 15, "schedule": {"enabled": True}})
    got = c.load()
    assert got["post_s"] == 15 and got["schedule"]["enabled"] and got["schedule"]["windows"]
    (tmp_path / "ctl.json").write_text("{not json")
    assert c.load() == c.normalize({})  # corrupt file never crashes the service
    c.pause(30, datetime(2026, 1, 1, tzinfo=timezone.utc))
    assert c.load()["paused_until"].startswith("2026-01-01T00:30")
    c.pause(0)
    assert c.load()["paused_until"] is None


def test_preview_is_on_by_default_and_can_be_switched_off():
    assert c.normalize({})["preview"] is True
    assert c.normalize({"preview": False})["preview"] is False
    assert c.normalize({"preview": "no"})["preview"] is True  # not a boolean: ignored


def test_rotation_accepts_only_quarter_turns():
    assert c.normalize({"rotate": 90})["rotate"] == 90
    for bad in (45, "90", True, None, 360, -90):
        assert c.normalize({"rotate": bad})["rotate"] == c.DEFAULTS["rotate"], bad
    assert c.normalize({"rotate": 270}, base=c.normalize({"rotate": 90}))["rotate"] == 270
    assert c.normalize({}, base=c.normalize({"rotate": 180}))["rotate"] == 180  # a patch without it keeps the saved value


def test_preview_request_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr(c, "REQUEST_FILE", tmp_path / "req.json")
    assert c.pending_request() is None
    a = c.request_preview()
    assert c.pending_request() == a
    b = c.request_preview()
    assert b != a and c.pending_request() == b
    (tmp_path / "req.json").write_text("garbage")
    assert c.pending_request() is None
