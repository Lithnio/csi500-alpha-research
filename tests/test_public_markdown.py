from __future__ import annotations

import re
from pathlib import Path
from urllib.parse import unquote

import pytest

REPOSITORY = Path(__file__).resolve().parents[1]
PUBLIC_MARKDOWN = (
    REPOSITORY / "README.md",
    REPOSITORY / "TECHNICAL.md",
    REPOSITORY / "docs" / "archive" / "README.v1.md",
)


@pytest.mark.parametrize("path", PUBLIC_MARKDOWN, ids=lambda path: path.name)
def test_public_markdown_uses_github_safe_math(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert "$$" not in text
    assert "\\operatorname" not in text

    prose_lines: list[str] = []
    fence: str | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("```"):
            fence = (stripped[3:].strip() or "code") if fence is None else None
            continue
        if fence is None:
            prose_lines.append(line)
    assert fence is None, f"unclosed Markdown fence in {path}"

    prose = "\n".join(prose_lines)
    prose_without_robust_inline_math = re.sub(r"\$`[^`\n]+`\$", "", prose)
    assert "$" not in prose_without_robust_inline_math


@pytest.mark.parametrize("path", PUBLIC_MARKDOWN, ids=lambda path: path.name)
def test_public_markdown_local_links_exist(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    targets = re.findall(r"!?\[[^\]]*\]\(([^)]+)\)", text)
    for raw_target in targets:
        target = raw_target.strip().strip("<>")
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        relative = unquote(target.split("#", maxsplit=1)[0])
        resolved = (path.parent / relative).resolve()
        assert resolved.exists(), f"missing local Markdown target: {target} in {path}"
