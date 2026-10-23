from pathlib import Path


POWER_HTML = Path(__file__).parents[1].joinpath("blinkwatch", "power.html").read_text()


def test_power_dashboard_preserves_month_and_tariff_controls():
    assert 'id="month"' in POWER_HTML
    assert 'id="price"' in POWER_HTML
    assert 'id="price" type="number" min="0" max="10" step="0.01" inputmode="decimal" value="0.28"' in POWER_HTML


def test_power_dashboard_exposes_live_and_monthly_summary_regions():
    assert 'id="live-power"' in POWER_HTML
    assert 'id="metric-energy"' in POWER_HTML
    assert 'id="metric-measured"' in POWER_HTML


def test_power_dashboard_has_accessible_device_history_and_settings_navigation():
    assert 'id="devices"' in POWER_HTML
    assert 'id="history"' in POWER_HTML
    assert 'id="settings"' in POWER_HTML
    assert 'aria-label="Navigazione principale"' in POWER_HTML


def test_power_dashboard_has_dark_theme_and_detail_routes():
    assert 'data-theme="dark"' in POWER_HTML
    assert "#/pc/" in POWER_HTML and "#/net/" in POWER_HTML


def test_guess_kind_uses_first_hostname_label_only():
    from blinkwatch.power import guess_kind
    assert guess_kind("Redmi-Pad-SE.homenet.telecomitalia.it", "aa:bb:cc:dd:ee:ff", "192.168.1.23") == "tablet"
    assert guess_kind("Host-007.homenet.telecomitalia.it", "34:7d:e4:46:b0:32", "192.168.1.25") == "other"
    assert guess_kind("", "4e:6a:07:7d:04:c9", "192.168.1.29") == "phone"  # private MAC, no name
