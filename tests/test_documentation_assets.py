"""Integrity checks for public documentation and figure assets."""

from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote


REPO_ROOT = Path(__file__).resolve().parents[1]
DOCUMENTS = (
    REPO_ROOT / "README.md",
    REPO_ROOT / "docs" / "RESULTS_GALLERY.md",
)
RESULTS_ROOT = REPO_ROOT / "docs" / "assets" / "results"

MARKDOWN_TARGET = re.compile(
    r"!?\[[^\]]*\]\(([^)\s]+)(?:\s+['\"][^)]*['\"])?\)"
)
HTML_IMAGE_SOURCE = re.compile(
    r"<img\b[^>]*\bsrc=['\"]([^'\"]+)['\"]",
    re.IGNORECASE,
)


def _local_references(document: Path) -> list[Path]:
    """Return normalized local targets referenced by one Markdown document."""

    text = document.read_text(encoding="utf-8")
    targets = [
        *MARKDOWN_TARGET.findall(text),
        *HTML_IMAGE_SOURCE.findall(text),
    ]
    references: list[Path] = []
    for target in targets:
        normalized = unquote(target.strip().split("#", maxsplit=1)[0])
        if not normalized or normalized.startswith(
            ("#", "http://", "https://", "mailto:")
        ):
            continue
        references.append((document.parent / normalized).resolve())
    return references


def _relative_paths(paths: set[Path] | list[Path]) -> list[str]:
    return sorted(str(path.relative_to(REPO_ROOT)) for path in paths)


def test_documentation_local_references_resolve() -> None:
    missing = {
        reference
        for document in DOCUMENTS
        for reference in _local_references(document)
        if not reference.exists()
    }

    assert not missing, "Missing documentation targets:\n" + "\n".join(
        _relative_paths(missing)
    )


def test_results_gallery_covers_every_public_figure() -> None:
    gallery = REPO_ROOT / "docs" / "RESULTS_GALLERY.md"
    referenced = {
        reference
        for reference in _local_references(gallery)
        if reference.suffix.lower() == ".png" and RESULTS_ROOT in reference.parents
    }
    available = set(RESULTS_ROOT.rglob("*.png"))

    assert referenced == available, (
        "The results gallery and public figure tree differ.\n"
        f"Unreferenced figures: {_relative_paths(available - referenced)}\n"
        f"Missing figures: {_relative_paths(referenced - available)}"
    )


def test_readme_exposes_framework_and_results_gallery() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    assert "docs/assets/framework/naformer-framework.png" in readme
    assert "docs/RESULTS_GALLERY.md" in readme
