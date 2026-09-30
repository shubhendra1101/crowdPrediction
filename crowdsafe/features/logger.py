"""Feature logging: Parquet for experiments (one file per clip / run), SQLite for the live demo."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from crowdsafe.features.schema import conform, validate


def write_parquet(df: pd.DataFrame, path: Path) -> Path:
    """Validate and write one clip/run's feature table to Parquet."""
    df = conform(df)
    problems = validate(df)
    if problems:
        raise ValueError(f"Feature table invalid: {problems}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, index=False)
    return path


def read_features(paths: list[Path] | Path) -> pd.DataFrame:
    """Read one Parquet file, a folder of them, or a list, into one table."""
    if isinstance(paths, (str, Path)) and Path(paths).is_dir():
        paths = sorted(Path(paths).rglob("*.parquet"))
    elif isinstance(paths, (str, Path)):
        paths = [Path(paths)]
    frames = [pd.read_parquet(p) for p in paths]
    return conform(pd.concat(frames, ignore_index=True)) if frames else conform(pd.DataFrame())


class SqliteLogger:
    """Append feature rows to a SQLite table (live demo). One row per zone per second."""

    def __init__(self, path: Path, table: str = "features") -> None:
        self.path, self.table = Path(path), table
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.con = sqlite3.connect(self.path)

    def append(self, df: pd.DataFrame) -> None:
        """Validate and append rows."""
        df = conform(df)
        problems = validate(df)
        if problems:
            raise ValueError(f"Feature rows invalid: {problems}")
        df.to_sql(self.table, self.con, if_exists="append", index=False)

    def latest(self, camera_id: str, seconds: float) -> pd.DataFrame:
        """Rows of one camera from the last ``seconds`` seconds of logged time."""
        q = (f"SELECT * FROM {self.table} WHERE camera_id = ? AND timestamp >= "
             f"(SELECT MAX(timestamp) FROM {self.table} WHERE camera_id = ?) - ? ORDER BY timestamp")
        return pd.read_sql_query(q, self.con, params=(camera_id, camera_id, seconds))

    def close(self) -> None:
        self.con.close()
