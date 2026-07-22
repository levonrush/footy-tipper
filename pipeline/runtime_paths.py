"""Single source of truth for overridable runtime artifact paths.

Normal operator and Actions runs use the repository's ``data/`` and ``models/``
directories.  The local model updater sets explicit paths so training and
validation happen in isolation and cannot damage the active artifacts.
"""

from __future__ import annotations

import os
import pathlib


def project_root() -> pathlib.Path:
    override = os.getenv("FOOTY_TIPPER_PROJECT_ROOT")
    return pathlib.Path(override).expanduser().resolve() if override else pathlib.Path.cwd()


def database_path(root: pathlib.Path | None = None) -> pathlib.Path:
    override = os.getenv("FOOTY_TIPPER_DB_PATH")
    if override:
        return pathlib.Path(override).expanduser().resolve()
    base = pathlib.Path(root) if root is not None else project_root()
    return base / "data" / "footy-tipper-db.sqlite"


def models_path(root: pathlib.Path | None = None) -> pathlib.Path:
    override = os.getenv("FOOTY_TIPPER_MODELS_DIR")
    if override:
        return pathlib.Path(override).expanduser().resolve()
    base = pathlib.Path(root) if root is not None else project_root()
    return base / "models"
