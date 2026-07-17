from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import yaml


ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "docs" / "knowledge"
FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
MARKDOWN_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def _metadata(path: Path) -> dict[str, object]:
    match = FRONTMATTER.match(path.read_text(encoding="utf-8"))
    assert match, f"{path.relative_to(ROOT)} is missing YAML frontmatter"
    value = yaml.safe_load(match.group(1))
    assert isinstance(value, dict), f"{path.relative_to(ROOT)} frontmatter is not a mapping"
    return value


def test_okf_bundle_has_reserved_entrypoints() -> None:
    assert (BUNDLE / "index.md").is_file()
    assert (BUNDLE / "log.md").is_file()
    assert _metadata(BUNDLE / "index.md")["okf_version"] == "0.1"


def test_okf_concept_files_have_nonempty_type() -> None:
    concept_files = sorted(BUNDLE.glob("*.md"))
    assert len(concept_files) >= 10
    for path in concept_files:
        if path.name in {"index.md", "log.md"}:
            continue
        metadata = _metadata(path)
        assert isinstance(metadata.get("type"), str)
        assert str(metadata["type"]).strip(), f"{path.relative_to(ROOT)} has an empty type"


def _assert_local_links_resolve(paths: list[Path]) -> None:
    for path in paths:
        text = path.read_text(encoding="utf-8")
        for raw_target in MARKDOWN_LINK.findall(text):
            target = raw_target.strip().strip("<>").split("#", 1)[0]
            if not target or target.startswith(("#", "http://", "https://", "mailto:")):
                continue
            resolved = (path.parent / unquote(target)).resolve()
            assert resolved.exists(), (
                f"broken link in {path.relative_to(ROOT)}: {raw_target}"
            )


def test_local_markdown_links_resolve() -> None:
    _assert_local_links_resolve(sorted(BUNDLE.glob("*.md")))


def test_handover_entrypoint_links_resolve() -> None:
    _assert_local_links_resolve(
        [ROOT / "README.md", ROOT / "AGENTS.md", ROOT / "HANDOVER.md", ROOT / "docs" / "README.md"]
    )


def test_knowledge_log_uses_iso_date_headings() -> None:
    text = (BUNDLE / "log.md").read_text(encoding="utf-8")
    headings = re.findall(r"^## (\d{4}-\d{2}-\d{2})$", text, flags=re.MULTILINE)
    assert headings
    assert headings == sorted(headings, reverse=True)
