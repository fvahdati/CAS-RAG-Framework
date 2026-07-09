"""Tests for timezone-aware datetime helpers."""

from datetime import timezone

from cas_rag.core.datetime_utils import utc_now, utc_now_iso


def test_utc_now_is_timezone_aware():
    now = utc_now()
    assert now.tzinfo == timezone.utc


def test_utc_now_iso_contains_offset():
    iso = utc_now_iso()
    assert iso.endswith("+00:00") or iso.endswith("Z")
