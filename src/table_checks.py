#### table_checks.py
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import pandas as pd


def repository_root() -> Path:
	#### Return the public repository root.
	return Path(__file__).resolve().parents[1]


def derived_dir() -> Path:
	#### Return the released aggregate-table directory.
	return repository_root() / "data" / "derived"


def read_derived_csv(filename: str) -> pd.DataFrame:
	#### Read one aggregate table from data/derived.
	path = derived_dir() / filename
	if not path.is_file():
		raise FileNotFoundError(f"Missing derived table: {filename}")
	return pd.read_csv(path)


def round_half_up(value: float, digits: int) -> float:
	#### Round half away from zero to the manuscript number of decimal places.
	if digits == 0:
		quantiser = Decimal("1")
	else:
		quantiser = Decimal("0." + ("0" * (digits - 1)) + "1")
	rounded = Decimal(str(value)).quantize(quantiser, rounding=ROUND_HALF_UP)
	if digits == 0:
		return int(rounded)
	return float(rounded)


def require_columns(frame: pd.DataFrame, columns: set[str], table_name: str) -> None:
	#### Stop if a released table is missing a column the check needs.
	missing = sorted(columns.difference(frame.columns))
	if missing:
		raise ValueError(f"{table_name} is missing columns: {missing}")


def report_mismatches(mismatches: list[str], table_name: str) -> None:
	#### Raise when rounded CSV values differ from the manuscript table.
	if mismatches:
		detail = "\n".join(mismatches)
		raise ValueError(f"{table_name} does not match the manuscript:\n{detail}")
