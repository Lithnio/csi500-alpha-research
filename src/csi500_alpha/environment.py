from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

from csi500_alpha.errors import CredentialError


def load_project_environment(project_root: Path) -> None:
    """Load local secrets without overriding process-level configuration."""
    load_dotenv(project_root / ".env", override=False)


def require_token(variable_name: str) -> str:
    token = os.environ.get(variable_name, "").strip()
    if not token:
        raise CredentialError(
            f"{variable_name} is not configured. Copy .env.example to .env and fill it locally."
        )
    return token

