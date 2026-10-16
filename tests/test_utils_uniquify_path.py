"""Tests for API/utils.uniquify_path and its callers.

``uniquify_path`` documents itself as "Return a unique path if target already
exists", but its body was ``return target``, so every caller that relied on it
to avoid clobbering an existing file silently overwrote that file instead.

The API modules use flat imports (``from config import ...``), so ``API/`` has to
be on ``sys.path`` before ``utils`` can be imported.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "API"))

from utils import save_markdown_report, uniquify_path  # noqa: E402


def test_returns_target_unchanged_when_missing(tmp_path):
    """The common case must stay a no-op so callers keep the nice name."""
    target = tmp_path / "report.md"
    assert uniquify_path(target) == target


def test_returns_a_different_path_when_target_exists(tmp_path):
    """Core contract: an existing target must never be handed back."""
    target = tmp_path / "report.md"
    target.write_text("original", encoding="utf-8")

    assert uniquify_path(target) != target


def test_collision_keeps_parent_and_suffix(tmp_path):
    """Renaming must not move the file or drop its extension."""
    target = tmp_path / "data.csv"
    target.write_text("original", encoding="utf-8")

    result = uniquify_path(target)

    assert result.parent == tmp_path
    assert result.suffix == ".csv"


def test_skips_every_already_taken_collision(tmp_path):
    """All taken "name (n).ext" slots must be stepped over."""
    target = tmp_path / "report.md"
    target.write_text("original", encoding="utf-8")
    (tmp_path / "report (1).md").write_text("one", encoding="utf-8")
    (tmp_path / "report (2).md").write_text("two", encoding="utf-8")

    result = uniquify_path(target)

    assert not result.exists()
    assert result.name == "report (3).md"


def test_continues_counting_from_an_already_suffixed_stem(tmp_path):
    """Feeding back a previously uniquified name must not restart at 1."""
    target = tmp_path / "report (2).md"
    target.write_text("two", encoding="utf-8")
    (tmp_path / "report (3).md").write_text("three", encoding="utf-8")

    result = uniquify_path(target)

    assert not result.exists()
    assert result.name == "report (4).md"


def test_handles_a_suffixless_name(tmp_path):
    """Files without an extension still need a usable collision name."""
    target = tmp_path / "README"
    target.write_text("original", encoding="utf-8")

    result = uniquify_path(target)

    assert not result.exists()
    assert result.name == "README (1)"


def test_save_markdown_report_does_not_overwrite_an_earlier_report(tmp_path):
    """Two reports saved under the same name must both survive."""
    base_name = "Conversation_Report_20260101_000000"

    first = save_markdown_report("first report", base_name, tmp_path)
    second = save_markdown_report("second report", base_name, tmp_path)

    assert first != second
    assert first.read_text(encoding="utf-8") == "first report"
    assert second.read_text(encoding="utf-8") == "second report"
