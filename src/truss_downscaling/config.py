from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    raw: dict[str, Any]
    path: Path

    @property
    def run(self) -> dict[str, Any]: return self.raw.setdefault("run", {})
    @property
    def data(self) -> dict[str, Any]: return self.raw.setdefault("data", {})
    @property
    def periods(self) -> dict[str, Any]: return self.raw.setdefault("periods", {})
    @property
    def model(self) -> dict[str, Any]: return self.raw.setdefault("model", {})
    @property
    def training(self) -> dict[str, Any]: return self.raw.setdefault("training", {})
    @property
    def scenario(self) -> dict[str, Any]: return self.raw.setdefault("scenario", {})

    @property
    def output_root(self) -> Path:
        return Path(self.run.get("output_root", "runs")) / self.run.get("id", self.path.stem)

    def model_class(self):
        path = self.model.get("class_path", "truss_downscaling.models.residual_unet.ResidualUNet")
        module, name = path.rsplit(".", 1)
        return getattr(importlib.import_module(module), name)


def load_config(path: str | Path) -> Config:
    path = Path(path).resolve()
    with path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}
    config = Config(raw=raw, path=path)
    if len(config.data.get("input_channels", [])) != 3 or len(config.data.get("target_channels", [])) != 3:
        raise ValueError("the current downscaling contract requires three input and three target channels")
    config.model_class()
    return config
