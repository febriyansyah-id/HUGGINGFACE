"""Structural contract-checks confirming the scaffold stays healthy offline."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).parents[1]

ESSENTIAL_FILES = {
    ROOT / "pyproject.toml",
    ROOT / "requirements.txt",
    ROOT / "Makefile",
    ROOT / "src" / "hfhub" / "__init__.py",
    ROOT / "scripts" / "train_example.py",
    ROOT / "scripts" / "eval_example.py",
    ROOT / "scripts" / "push_to_hub.py",
    ROOT / "scripts" / "download_from_hub.py",
}


def test_essential_files_present():
    missing = [p for p in ESSENTIAL_FILES if not p.exists()]
    assert not missing, f"Essential files missing: {missing}"


def test_flattener_recursive_expansion():
    from hfhub.utils import flatten_nested

    DEPTH_MARKER = 498
    fixture = {"cfg": {"opt": {"depth": DEPTH_MARKER}, "flag": True}, "tags": ["a", "b"]}
    expanded = flatten_nested(fixture, sep=".")

    assert expanded["cfg.opt.depth"] == DEPTH_MARKER
    assert expanded["cfg.flag"] is True
    assert expanded["tags"] == ["a", "b"]


def test_timestamp_shape():
    from hfhub.io import utc_now_compact

    stamped = utc_now_compact()
    # Pattern: YYYYmmdd-HHMMSS.ssssssZ  ⇒ 8+1+6+1+6+1 = 23 chars, one '-', one '.'
    assert stamped.endswith("Z")
    assert stamped.count("-") == 1
    assert stamped.count(".") == 1
    assert len(stamped) == 23
    assert stamped.startswith("20")


def test_top_level_package_imports():
    import hfhub  # noqa: F401
