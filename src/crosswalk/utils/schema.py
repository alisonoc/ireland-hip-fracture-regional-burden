#### schema.py

import json
from pathlib import Path

import pandas as pd


def load_schema(path: Path) -> dict:
	#### Load dataset schema from JSON
	with open(path, "r", encoding="utf-8") as file:
		return json.load(file)


def apply_schema(
	df: pd.DataFrame,
	schema: dict,
	preserve_category_codes: bool = False,
) -> pd.DataFrame:
	#### Apply manual schema dtypes to dataframe columns
	df = df.copy()

	for column, spec in schema["columns"].items():
		if column not in df.columns:
			if spec.get("required", False):
				raise ValueError(f"Required column missing: {column}")
			continue

		dtype = spec.get("dtype", spec.get("type"))

		if dtype in {"datetime", "datetime64"}:
			dayfirst = spec.get("dayfirst", False)
			df[column] = pd.to_datetime(
				df[column],
				errors="coerce",
				dayfirst=dayfirst,
			)

		elif dtype in {"integer", "int", "int64"}:
			df[column] = pd.to_numeric(
				df[column],
				errors="coerce",
			).astype("Int64")

		elif dtype == "category":
			if "map" in spec:
				codes = pd.to_numeric(
					df[column],
					errors="coerce",
				).astype("Int64")

				mapping = {
					int(key): value
					for key, value in spec["map"].items()
				}

				if preserve_category_codes:
					df[f"{column}_code"] = codes

				df[column] = codes.map(mapping).astype("category")

			else:
				df[column] = df[column].astype("category")

	return df