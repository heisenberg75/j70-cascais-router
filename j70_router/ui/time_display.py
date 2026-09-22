"""User-facing time formatting for the Cascais race area."""

from __future__ import annotations

from datetime import datetime
from zoneinfo import ZoneInfo


CASCAIS_TIMEZONE = ZoneInfo("Europe/Lisbon")


def cascais_local_time(value: datetime) -> datetime:
    """Convert an aware instant to Portugal mainland civil time."""

    if value.tzinfo is None:
        raise ValueError("display datetimes must be timezone-aware")
    return value.astimezone(CASCAIS_TIMEZONE)


def format_cascais_time(value: datetime) -> str:
    """Format an instant with its DST-aware WET/WEST abbreviation."""

    return cascais_local_time(value).strftime("%Y-%m-%d %H:%M %Z")
