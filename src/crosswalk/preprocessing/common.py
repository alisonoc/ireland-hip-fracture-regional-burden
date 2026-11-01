#### common.py

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

from crosswalk.preprocessing.reference import harmonize_hse_region_series, order_hse_regions
from crosswalk.utils.io import read_required_csv
from crosswalk.utils.validation import require_columns


GEO_COUNTY_TO_REGION_BASELINE_FILE = "geo_county_to_region_baseline.csv"
CSO_POPULATION_MASTER_FILE = "cso_population_master_age_sex_region_county_year.csv"


def flatten_grouped_schema(schema: dict,
						   group_names: list[str],
						   ) -> dict:
	#### Flatten grouped schema columns into the columns key expected by apply_schema.
	ret = schema.copy()
	columns = {}

	for group_name in group_names:
		group = ret.get(group_name, {})

		if not isinstance(group, dict):
			raise ValueError(f"Schema group must be a dictionary: {group_name}")

		overlap = set(columns) & set(group)

		if overlap:
			raise ValueError(f"Schema columns appear in more than one group: {sorted(overlap)}")

		columns.update(group)

	if not columns:
		raise ValueError("Grouped schema contains no columns after flattening.")

	ret["columns"] = columns

	return ret


def get_schema_column_names(schema: dict,
							group_name: str,
							) -> list[str]:
	#### Return column names from a grouped schema dictionary.
	group = schema.get(group_name)

	if group is None:
		raise ValueError(f"Schema missing column group: {group_name}")

	if not isinstance(group, dict):
		raise ValueError(f"Schema column group must be a dictionary: {group_name}")

	ret = list(group.keys())

	if not ret:
		raise ValueError(f"Schema column group is empty: {group_name}")

	return ret


def get_schema_derived_mapping(schema: dict,
							   mapping_name: str,
							   ) -> dict:
	#### Return a named derived mapping from a schema dictionary.
	mappings = schema.get("derived_mappings", {})

	if mapping_name not in mappings:
		raise ValueError(f"Schema missing derived mapping: {mapping_name}")

	ret = mappings[mapping_name]

	if not isinstance(ret, dict):
		raise ValueError(f"Schema derived mapping must be a dictionary: {mapping_name}")

	if not ret:
		raise ValueError(f"Schema derived mapping is empty: {mapping_name}")

	return ret


def normalise_county_label(series: pd.Series) -> pd.Series:
	#### Normalise county labels for preprocessing-stage county joins.
	ret = series.astype("string").str.strip().str.lower()

	ret = ret.apply(lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("utf-8")
					if pd.notna(x) else x)

	ret = ret.str.replace("co. ", "", regex=False)
	ret = ret.str.replace("county ", "", regex=False)
	ret = ret.str.replace(r"\s+", " ", regex=True).str.strip()

	return ret


def load_valid_hse_regions(dirs: dict[str, Path]) -> list[str]:
	#### Load canonical HSE region names from the geography baseline output.
	path = dirs["processed"] / GEO_COUNTY_TO_REGION_BASELINE_FILE

	geo = read_required_csv(path=path,
							name="geography county-to-region baseline",)

	require_columns(df=geo,
					required={"hse_region"},
					name="geography county-to-region baseline",)

	ret = harmonize_hse_region_series(geo["hse_region"].dropna().astype("string").str.strip())
	ret = order_hse_regions(ret.drop_duplicates())

	return ret


def load_valid_cso_age_groups(dirs: dict[str, Path]) -> list[str]:
	#### Load canonical CSO age-group labels from the processed CSO population file.
	path = dirs["processed"] / CSO_POPULATION_MASTER_FILE

	cso = read_required_csv(path=path,
							name="CSO population master",)

	require_columns(df=cso,
					required={"age_group"},
					name="CSO population master",)

	ret = (cso["age_group"]
		   .dropna()
		   .astype("string")
		   .str.strip()
		   .drop_duplicates()
		   .sort_values()
		   .to_list())

	return ret


def collapse_county_to_hse_lookup(county_to_region: pd.DataFrame) -> pd.DataFrame:
	#### Collapse county-to-HSE lookup to one row per county and HSE region.
	required = {"county_english",
				"hse_region",
				"small_area_weight",}

	require_columns(df=county_to_region,
					required=required,
					name="county-to-HSE lookup",)

	df = county_to_region.copy()
	df["county_english"] = normalise_county_label(df["county_english"])
	df["hse_region"] = df["hse_region"].astype("string").str.strip()
	df["small_area_weight"] = pd.to_numeric(df["small_area_weight"], errors="coerce")

	if df["county_english"].isna().any():
		raise ValueError("County-to-HSE lookup contains missing county_english values.")

	if df["hse_region"].isna().any():
		raise ValueError("County-to-HSE lookup contains missing hse_region values.")

	if df["small_area_weight"].isna().any():
		raise ValueError("County-to-HSE lookup contains missing or invalid small_area_weight values.")

	ret = (df.groupby(["county_english", "hse_region"],
					  as_index=False,
					  dropna=False,
					  observed=True,)
		   .agg(small_area_weight=("small_area_weight", "sum"),))

	weight_check = (ret.groupby("county_english",
								as_index=False,
								dropna=False,
								observed=True,)
					.agg(weight_sum=("small_area_weight", "sum"),))

	bad_weights = weight_check.loc[~weight_check["weight_sum"].round(6).eq(1)].copy()

	if not bad_weights.empty:
		raise ValueError(f"Collapsed county-to-HSE weights do not sum to 1: {bad_weights.to_dict(orient='records')}")

	return ret