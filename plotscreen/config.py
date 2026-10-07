"""Run configuration, loaded from a YAML file."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field, field_validator

Role = Literal["train", "eval", "transfer"]


class Aoi(BaseModel):
    """A lon/lat bounding box with a role in the experiment."""

    name: str
    bbox: tuple[float, float, float, float]
    role: Role

    @field_validator("bbox")
    @classmethod
    def _ordered(cls, v: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        min_lon, min_lat, max_lon, max_lat = v
        if not (min_lon < max_lon and min_lat < max_lat):
            raise ValueError("bbox must be (min_lon, min_lat, max_lon, max_lat)")
        return v


class Settings(BaseModel):
    before_year: int = 2020
    after_year: int = 2025
    hansen_version: str = "GFC-2025-v1.13"
    min_tree_cover: int = Field(30, ge=1, le=100)
    plot_px: int = Field(20, ge=2)
    min_loss_px: int = Field(50, ge=1)
    top_k: int = Field(50, ge=1)
    alpha: float = Field(0.05, gt=0, lt=1)
    beta: float = Field(0.05, gt=0, lt=1)
    confidence: float | None = Field(None, gt=0, lt=1)
    cluster_correction: bool = True
    aois: list[Aoi] = []

    @property
    def cutoff_code(self) -> int:
        """Hansen encodes the loss year as year minus 2000."""
        return self.before_year - 2000

    def by_role(self, role: Role) -> list[Aoi]:
        return [a for a in self.aois if a.role == role]


def load_settings(path: str | Path = "aois.yaml") -> Settings:
    with open(path, encoding="utf-8") as fh:
        return Settings(**yaml.safe_load(fh))


def data_dir() -> Path:
    """Where downloaded cubes and labels are cached. Not committed."""
    root = Path(os.environ.get("PLOTSCREEN_DATA_DIR", "data"))
    root.mkdir(parents=True, exist_ok=True)
    return root
