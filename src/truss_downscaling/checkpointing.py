from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import torch


def atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_json(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    os.replace(temporary, path)


def mark_success(stage: Path, payload: dict[str, Any]) -> None:
    save_json(payload, stage / "manifest.json")
    (stage / "_SUCCESS").write_text("ok\n", encoding="utf-8")


def stage_is_complete(stage: Path) -> bool:
    return (stage / "_SUCCESS").exists() and (stage / "manifest.json").exists()
