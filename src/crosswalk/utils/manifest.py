#### manifest.py

import json
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

import pandas as pd

from crosswalk.utils.logging import json_default


def dataframe_manifest(
	df: pd.DataFrame,
	output_path: Path,
	stage: str,
	tz: str = "Europe/Dublin",
) -> dict:
	#### Build a compact manifest for a dataframe checkpoint
	manifest = {
		"stage": stage,
		"created_at": datetime.now(ZoneInfo(tz)).isoformat(),
		"output_path": str(output_path),
		"shape": list(df.shape),
		"columns": list(df.columns),
		"missing_values": df.isna().sum().to_dict(),
	}

	if "date" in df.columns:
		dates = pd.to_datetime(df["date"], errors="coerce")
		manifest["date_min"] = str(dates.min())
		manifest["date_max"] = str(dates.max())

	if "year" in df.columns:
		manifest["year_min"] = int(df["year"].min())
		manifest["year_max"] = int(df["year"].max())

	return manifest


def write_manifest(manifest: dict, path: Path) -> Path:
	#### Write manifest dictionary to JSON
	path.parent.mkdir(parents=True, exist_ok=True)

	with open(path, "w", encoding="utf-8") as file:
		json.dump(
			manifest,
			file,
			indent=2,
			ensure_ascii=False,
			default=json_default,
		)

	return path

def column_audit(df: pd.DataFrame) -> pd.DataFrame:
	#### Create column-level audit with dtype and missingness
	n_rows = len(df)

	audit = pd.DataFrame({
		"column": df.columns,
		"dtype": [str(df[col].dtype) for col in df.columns],
		"n_missing": [int(df[col].isna().sum()) for col in df.columns],
		"pct_missing": [
			round(float(df[col].isna().mean() * 100), 3)
			for col in df.columns
		],
		"n_unique": [
			int(df[col].nunique(dropna=True))
			for col in df.columns
		],
	})

	audit["n_rows"] = n_rows

	return audit

def write_column_audit(df: pd.DataFrame, output_path: Path) -> Path:
	#### Write column-level audit to CSV
	output_path.parent.mkdir(parents=True, exist_ok=True)

	audit = column_audit(df)
	audit.to_csv(output_path, index=False)

	return output_path