#!/usr/bin/env python3
"""
Satellite Watch — orbital state report.

For each satellite on the watchlist, fetches the current TLE from CelesTrak,
propagates it to now with SGP4, and reports:

  * exact position  — sub-satellite latitude/longitude, altitude
  * inclination     — plus the rest of the orbital elements
  * movement        — what changed since the previous report (manoeuvre detection)
  * speed           — inertial orbital velocity, and orbital period

Run state is written to disk for the workflow to pick up:
  skip.flag    written when it is not a scheduled Sydney hour (nothing else is)
  report.txt   the email body
  subject.txt  the email subject line
  report.json  machine-readable copy of this run
  history/     one JSON file per satellite, carried forward between runs
"""

from __future__ import annotations

import json
import math
import os
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
from sgp4.api import Satrec, jday

# ---------------------------------------------------------------------------
# Watchlist — edit this. Add or remove lines freely.
# Find a NORAD catalogue number at https://celestrak.org/satcat/search.php
# ---------------------------------------------------------------------------

WATCHLIST = [
    (25544, "ISS (ZARYA)"),
    (48274, "CSS (TIANHE)"),
    (20580, "HST"),
    (43013, "NOAA-20"),
    (40146, "OPTUS 10"),
]

# Local time gate. The workflow fires four times a day in UTC; the report is
# only produced when the Sydney clock reads one of these hours.
LOCAL_TZ = ZoneInfo("Australia/Sydney")
REPORT_HOURS = (7, 17)

# Manoeuvre thresholds. A change larger than any of these between consecutive
# reports is flagged rather than treated as ordinary drift.
THRESH_SMA_KM = 0.5          # semi-major axis
THRESH_INC_DEG = 0.01        # inclination
THRESH_ECC = 0.0005          # eccentricity
THRESH_RAAN_DEG = 0.20       # right ascension of ascending node

CELESTRAK = "https://celestrak.org/NORAD/elements/gp.php?CATNR={}&FORMAT=tle"

MU = 398600.4418             # Earth gravitational parameter, km^3/s^2
R_EARTH = 6378.137           # WGS84 equatorial radius, km
F_EARTH = 1 / 298.257223563  # WGS84 flattening

HISTORY = Path("history")


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def gmst_rad(jd: float, fr: float) -> float:
    """Greenwich mean sidereal time, IAU 1982, radians."""
    tut1 = ((jd - 2451545.0) + fr) / 36525.0
    sec = (
        67310.54841
        + (876600.0 * 3600.0 + 8640184.812866) * tut1
        + 0.093104 * tut1 * tut1
        - 6.2e-6 * tut1 * tut1 * tut1
    )
    deg = (sec % 86400.0) / 240.0
    return math.radians(deg % 360.0)


def teme_to_ecef(r_teme, gmst: float):
    """Rotate a TEME position vector into an Earth-fixed frame."""
    c, s = math.cos(gmst), math.sin(gmst)
    x, y, z = r_teme
    return np.array([c * x + s * y, -s * x + c * y, z])


def ecef_to_geodetic(r_ecef):
    """WGS84 geodetic latitude (deg), longitude (deg), altitude (km)."""
    x, y, z = r_ecef
    lon = math.degrees(math.atan2(y, x))
    lon = (lon + 180.0) % 360.0 - 180.0

    e2 = F_EARTH * (2 - F_EARTH)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1 - e2))

    for _ in range(12):
        sin_lat = math.sin(lat)
        n = R_EARTH / math.sqrt(1 - e2 * sin_lat * sin_lat)
        alt = p / math.cos(lat) - n
        new_lat = math.atan2(z, p * (1 - e2 * n / (n + alt)))
        if abs(new_lat - lat) < 1e-12:
            lat = new_lat
            break
        lat = new_lat

    sin_lat = math.sin(lat)
    n = R_EARTH / math.sqrt(1 - e2 * sin_lat * sin_lat)
    alt = p / math.cos(lat) - n
    return math.degrees(lat), lon, alt


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def fetch_tle(catnr: int) -> tuple[str, str, str]:
    """Return (name, line1, line2) for a catalogue number."""
    req = urllib.request.Request(
        CELESTRAK.format(catnr),
        headers={"User-Agent": "satellite-watch (github actions)"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        text = resp.read().decode("utf-8", "replace")

    lines = [ln.rstrip() for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3 or lines[0].upper().startswith("NO GP DATA"):
        raise ValueError(f"no TLE returned for {catnr}")
    return lines[0].strip(), lines[1], lines[2]


def elements(sat: Satrec, name: str, catnr: int, now: datetime) -> dict:
    """Propagate to `now` and derive a full orbital state."""
    jd, fr = jday(
        now.year, now.month, now.day,
        now.hour, now.minute, now.second + now.microsecond / 1e6,
    )

    err, r, v = sat.sgp4(jd, fr)
    if err != 0:
        raise RuntimeError(f"SGP4 error {err} for {name}")

    r = np.array(r)
    v = np.array(v)

    lat, lon, alt = ecef_to_geodetic(teme_to_ecef(r, gmst_rad(jd, fr)))
    speed = float(np.linalg.norm(v))

    n_rad_s = sat.no_kozai / 60.0                       # mean motion
    a = (MU / (n_rad_s ** 2)) ** (1.0 / 3.0)            # semi-major axis, km
    period_min = 2 * math.pi / n_rad_s / 60.0
    ecc = sat.ecco

    epoch_jd = sat.jdsatepoch + sat.jdsatepochF
    epoch = datetime.fromtimestamp((epoch_jd - 2440587.5) * 86400.0, tz=timezone.utc)

    return {
        "catnr": catnr,
        "name": name,
        "utc": now.isoformat(timespec="seconds"),
        "lat_deg": lat,
        "lon_deg": lon,
        "alt_km": alt,
        "speed_km_s": speed,
        "inclination_deg": math.degrees(sat.inclo),
        "raan_deg": math.degrees(sat.nodeo) % 360.0,
        "arg_perigee_deg": math.degrees(sat.argpo) % 360.0,
        "eccentricity": ecc,
        "mean_anomaly_deg": math.degrees(sat.mo) % 360.0,
        "mean_motion_rev_day": n_rad_s * 86400.0 / (2 * math.pi),
        "sma_km": a,
        "period_min": period_min,
        "perigee_alt_km": a * (1 - ecc) - R_EARTH,
        "apogee_alt_km": a * (1 + ecc) - R_EARTH,
        "tle_epoch_utc": epoch.isoformat(timespec="seconds"),
        "tle_age_hours": (now - epoch).total_seconds() / 3600.0,
    }


# ---------------------------------------------------------------------------
# Movement
# ---------------------------------------------------------------------------

def compare(now_state: dict, prev: dict | None) -> dict | None:
    """Element deltas against the previous report, with a manoeuvre verdict."""
    if not prev:
        return None

    d = {
        "since_utc": prev.get("utc"),
        "d_sma_km": now_state["sma_km"] - prev["sma_km"],
        "d_inclination_deg": now_state["inclination_deg"] - prev["inclination_deg"],
        "d_eccentricity": now_state["eccentricity"] - prev["eccentricity"],
        "d_raan_deg": (now_state["raan_deg"] - prev["raan_deg"] + 180) % 360 - 180,
        "d_period_min": now_state["period_min"] - prev["period_min"],
    }

    flags = []
    if abs(d["d_sma_km"]) > THRESH_SMA_KM:
        flags.append(f"altitude change {d['d_sma_km']:+.2f} km")
    if abs(d["d_inclination_deg"]) > THRESH_INC_DEG:
        flags.append(f"inclination change {d['d_inclination_deg']:+.4f}°")
    if abs(d["d_eccentricity"]) > THRESH_ECC:
        flags.append(f"eccentricity change {d['d_eccentricity']:+.6f}")
    if abs(d["d_raan_deg"]) > THRESH_RAAN_DEG:
        flags.append(f"RAAN change {d['d_raan_deg']:+.3f}°")

    d["flags"] = flags
    d["manoeuvre_suspected"] = bool(flags)
    return d


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def hemi(lat: float, lon: float) -> str:
    ns = "N" if lat >= 0 else "S"
    ew = "E" if lon >= 0 else "W"
    return f"{abs(lat):6.2f}°{ns}  {abs(lon):7.2f}°{ew}"


def render(states: list[dict], errors: list[str], now_local: datetime) -> tuple[str, str]:
    moved = [s for s in states if s.get("delta", {}) and s["delta"].get("manoeuvre_suspected")]

    out = []
    out.append("SATELLITE WATCH")
    out.append(now_local.strftime("%A %d %B %Y, %H:%M %Z"))
    out.append(f"{now_local.astimezone(timezone.utc):%Y-%m-%d %H:%M} UTC")
    out.append("")

    if moved:
        out.append(f"** {len(moved)} object(s) showing orbit change since last report **")
        for s in moved:
            out.append(f"   {s['name']}: " + "; ".join(s["delta"]["flags"]))
        out.append("")

    out.append("=" * 68)

    for s in states:
        out.append("")
        out.append(f"{s['name']}   [{s['catnr']}]")
        out.append("-" * 68)
        out.append(f"  Position      {hemi(s['lat_deg'], s['lon_deg'])}   {s['alt_km']:8.1f} km altitude")
        out.append(f"  Speed         {s['speed_km_s']:.3f} km/s   ({s['speed_km_s'] * 3600:,.0f} km/h)")
        out.append(f"  Period        {s['period_min']:.2f} min   ({s['mean_motion_rev_day']:.4f} rev/day)")
        out.append("")
        out.append(f"  Inclination   {s['inclination_deg']:.4f}°")
        out.append(f"  RAAN          {s['raan_deg']:.4f}°")
        out.append(f"  Arg perigee   {s['arg_perigee_deg']:.4f}°")
        out.append(f"  Eccentricity  {s['eccentricity']:.7f}")
        out.append(f"  Mean anomaly  {s['mean_anomaly_deg']:.4f}°")
        out.append(f"  Semi-major    {s['sma_km']:.3f} km")
        out.append(f"  Perigee/Apo   {s['perigee_alt_km']:.1f} / {s['apogee_alt_km']:.1f} km")
        out.append("")

        d = s.get("delta")
        if not d:
            out.append("  Movement      no previous report — this is the baseline")
        else:
            since = d["since_utc"].replace("T", " ")[:16] if d["since_utc"] else "?"
            out.append(f"  Movement      since {since} UTC")
            out.append(f"     semi-major   {d['d_sma_km']:+.3f} km")
            out.append(f"     inclination  {d['d_inclination_deg']:+.5f}°")
            out.append(f"     eccentricity {d['d_eccentricity']:+.7f}")
            out.append(f"     RAAN         {d['d_raan_deg']:+.4f}°")
            out.append(f"     period       {d['d_period_min']:+.4f} min")
            if d["manoeuvre_suspected"]:
                out.append("     >> beyond drift thresholds — manoeuvre suspected")
            else:
                out.append("     consistent with natural drift")

        stale = "  << stale, treat position with caution" if s["tle_age_hours"] > 48 else ""
        out.append(f"  TLE epoch     {s['tle_epoch_utc'].replace('T', ' ')[:16]} UTC "
                   f"({s['tle_age_hours']:.1f} h old){stale}")

    if errors:
        out.append("")
        out.append("=" * 68)
        out.append("")
        out.append("NOT REPORTED")
        for e in errors:
            out.append(f"  {e}")

    out.append("")
    out.append("=" * 68)
    out.append("Elements from CelesTrak GP data, propagated with SGP4.")
    out.append("Positions are sub-satellite points on the WGS84 ellipsoid.")
    out.append("")

    if moved:
        subject = f"Satellite Watch — {len(moved)} orbit change(s) — {now_local:%d %b %H:%M}"
    elif errors and not states:
        subject = f"Satellite Watch — no data — {now_local:%d %b %H:%M}"
    else:
        subject = f"Satellite Watch — {len(states)} object(s) nominal — {now_local:%d %b %H:%M}"

    return subject, "\n".join(out)


# ---------------------------------------------------------------------------

def main() -> int:
    now_utc = datetime.now(timezone.utc)
    now_local = now_utc.astimezone(LOCAL_TZ)
    forced = bool(os.environ.get("FORCE_RUN"))

    if not forced and now_local.hour not in REPORT_HOURS:
        Path("skip.flag").write_text(
            f"Sydney time {now_local:%H:%M}; report hours are "
            f"{', '.join(f'{h:02d}:00' for h in REPORT_HOURS)}.\n"
        )
        print(f"Skipping: Sydney local time is {now_local:%H:%M %Z}.")
        return 0

    HISTORY.mkdir(exist_ok=True)

    states: list[dict] = []
    errors: list[str] = []

    for catnr, label in WATCHLIST:
        try:
            name, l1, l2 = fetch_tle(catnr)
            sat = Satrec.twoline2rv(l1, l2)
            state = elements(sat, name or label, catnr, now_utc)
        except (urllib.error.URLError, ValueError, RuntimeError, OSError) as exc:
            errors.append(f"{label} [{catnr}]: {exc}")
            print(f"  ! {label}: {exc}", file=sys.stderr)
            continue

        hist_file = HISTORY / f"{catnr}.json"
        prev = None
        if hist_file.exists():
            try:
                prev = json.loads(hist_file.read_text())
            except json.JSONDecodeError:
                prev = None

        state["delta"] = compare(state, prev)
        states.append(state)
        hist_file.write_text(json.dumps(state, indent=2) + "\n")
        print(f"  + {state['name']}")

    if not states and errors:
        # Every fetch failed — surface it rather than emailing an empty report.
        print("All fetches failed.", file=sys.stderr)

    subject, body = render(states, errors, now_local)

    Path("report.txt").write_text(body)
    Path("subject.txt").write_text(subject)
    Path("report.json").write_text(
        json.dumps(
            {
                "generated_utc": now_utc.isoformat(timespec="seconds"),
                "generated_local": now_local.isoformat(timespec="seconds"),
                "forced": forced,
                "satellites": states,
                "errors": errors,
            },
            indent=2,
        )
        + "\n"
    )

    print(f"\n{subject}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
