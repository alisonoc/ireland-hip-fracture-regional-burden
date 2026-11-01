#### preprocess_cso_projections.py

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from crosswalk.paths import (
	ProjectPaths,
	get_project_paths,
)
from crosswalk.cso_age_schema import (
	get_all_ages_label,
	validate_cso_age_group_schema,
)
from crosswalk.preprocessing.preprocess_cso import NUTS3_LABEL_MAPPING
from crosswalk.utils.io import (
	read_required_csv,
	read_required_json,
	save_dataset_with_manifest,
)
from crosswalk.utils.logging import (
	get_logger,
	log_block,
)
from crosswalk.utils.manifest import write_column_audit
from crosswalk.utils.validation import (
	coerce_numeric_columns,
	require_columns,
	validate_no_missing,
	validate_non_empty,
	validate_non_negative,
	validate_unique_rows,
)

CSO_AGE_GROUP_SCHEMA_FILE = "cso_age_groups_schema.json"

PEC26_PATTERN = "PEC26*.csv"
PEC28_PATTERN = "PEC28*.csv"

PEC26_AGE_OUTPUT_FILE = "cso_pec26_population_projection_age.csv"
PEC26_AGE_GROUP_OUTPUT_FILE = "cso_pec26_population_projection_age_group.csv"
PEC28_DEPENDENCY_OUTPUT_FILE = "cso_pec28_dependency_ratios.csv"

COUNTY_POPULATION_COMBINED_FILE = "cso_county_population_combined_pea04_pea08.csv"

COUNTY_GROWTH_AUDIT_OUTPUT_FILE = "cso_observed_county_growth_for_projection_audit.csv"
COUNTY_SHARE_CHANGE_AUDIT_OUTPUT_FILE = "cso_observed_county_share_change_for_projection_audit.csv"
COUNTY_SHARE_CHANGE_ALL_AGES_SUMMARY_OUTPUT_FILE = "cso_observed_county_share_change_all_ages_projection_summary.csv"

OBSERVED_COUNTY_POPULATION_SOURCE = "pea08_county_age_sex_observed"

IRELAND_GEOGRAPHY_LABEL = "Ireland"
NATIONAL_GEOGRAPHY_TYPE = "national"
NUTS3_GEOGRAPHY_TYPE = "nuts3"
PROJECTION_METHOD_PREFIX = "Method - "

PEC26_REQUIRED_COLUMNS = {"Statistic Label",
						  "Year",
						  "Age",
						  "Sex",
						  "Criteria for Projection",
						  "NUTS 3 Region",
						  "UNIT",
						  "VALUE",}

PEC28_REQUIRED_COLUMNS = {"Statistic Label",
						  "Year",
						  "Criteria for Projection",
						  "NUTS 3 Region",
						  "Dependents Status",
						  "VALUE",}

LOW_SHARE_CHANGE_THRESHOLD = 0.01
HIGH_SHARE_CHANGE_THRESHOLD = 0.03


def find_latest_raw_file(paths: ProjectPaths,
						 pattern: str,
						 ) -> Path:
	#### Find the latest matching CSO raw download under dataset/raw.
	matches = sorted(paths.raw.rglob(pattern),
					 key=lambda path: path.name)

	if not matches:
		raise FileNotFoundError(f"No raw file matching {pattern} found under {paths.raw}")

	return matches[-1]


def load_cso_age_group_schema(paths: ProjectPaths) -> dict:
	#### Load and validate the controlled CSO age-group schema.
	ret = read_required_json(path=paths.schemas / CSO_AGE_GROUP_SCHEMA_FILE,
							 name="CSO age-group schema",)

	validate_cso_age_group_schema(ret)

	return ret


def load_cso_projection_raw_files(paths: ProjectPaths) -> dict[str, pd.DataFrame]:
	#### Load the latest raw CSO PEC projection files from dataset/raw.
	pec26_path = find_latest_raw_file(paths=paths,
									  pattern=PEC26_PATTERN,)
	pec28_path = find_latest_raw_file(paths=paths,
									  pattern=PEC28_PATTERN,)

	ret = {"pec26": read_required_csv(path=pec26_path,
									  name=f"Raw CSO projection file {pec26_path.name}",),
		   "pec28": read_required_csv(path=pec28_path,
									  name=f"Raw CSO projection file {pec28_path.name}",),}

	return ret


def load_observed_county_population_for_projection_audit(dirs: dict[str, Path]) -> pd.DataFrame:
	#### Load observed county population rows used to audit projection allocation assumptions.
	#### This audit is intentionally located in the projection preprocessing module because it informs
	#### whether future NUTS3-to-county allocation should use static shares or a trend-adjusted method.
	#### It does not create future county projections.
	path = dirs["processed"] / COUNTY_POPULATION_COMBINED_FILE

	ret = read_required_csv(path=path,
	                        name="Observed county population for projection allocation audit", )

	required = {"year",
	            "age_group",
	            "sex",
	            "nuts3_names",
	            "county_english",
	            "county_population",
	            "population_source",
	            "is_interpolated", }

	require_columns(df=ret,
	                required=required,
	                name="Observed county population for projection allocation audit", )

	return ret


def parse_single_year_age(age_label: str) -> int | None:
	#### Parse a CSO projection age label into an integer age where possible.
	if pd.isna(age_label):
		return None

	age_label = str(age_label).strip()

	if age_label == "Under 1 year":
		return 0

	if age_label == "All ages":
		return None

	match = re.match(r"^(\d+)\s+year", age_label)

	if match is None:
		return None

	return int(match.group(1))


def parse_schema_age_bounds(age_group: str) -> tuple[int, int | None] | None:
	#### Parse a controlled CSO schema age-group label into lower and upper bounds.
	if pd.isna(age_group):
		return None

	age_group = str(age_group).strip()

	match = re.match(r"^(\d+)\s*-\s*(\d+)\s+years$", age_group)

	if match is not None:
		return int(match.group(1)), int(match.group(2))

	match = re.match(r"^(\d+)\s+years and over$", age_group)

	if match is not None:
		return int(match.group(1)), None

	return None


def build_single_age_to_schema_group_map(age_group_schema: dict,
										 max_age: int = 100,
										 ) -> dict[str, str]:
	#### Build a reusable mapping from CSO single-year age labels to controlled age groups.
	all_ages_label = get_all_ages_label(age_group_schema)
	ret = {"All ages": all_ages_label,
		   "Under 1 year": age_group_schema["age_group_order"][0],}

	for age in range(1, max_age + 1):
		for schema_age_group in age_group_schema["age_group_order"]:
			bounds = parse_schema_age_bounds(schema_age_group)

			if bounds is None:
				continue

			lower, upper = bounds

			if upper is None and age >= lower:
				ret[f"{age} years"] = schema_age_group
				ret[f"{age} year"] = schema_age_group
				break

			if upper is not None and lower <= age <= upper:
				ret[f"{age} years"] = schema_age_group
				ret[f"{age} year"] = schema_age_group
				break

	return ret


def normalise_projection_method(series: pd.Series) -> pd.Series:
	#### Normalise CSO projection method labels.
	ret = (series.astype("string")
		  .str.strip()
		  .str.replace(PROJECTION_METHOD_PREFIX, "", regex=False))

	return ret

def normalise_projection_age_label(age_label: str) -> str | pd.NA:
	#### Normalise raw CSO projection age labels to canonical single-year age labels.
	if pd.isna(age_label):
		return pd.NA

	ret = str(age_label).strip()

	if ret == "All ages":
		return "All ages"

	if ret == "Under 1 year":
		return "Under 1 year"

	match = re.match(r"^(\d+)\s+years?\s+and\s+over$", ret)

	if match is not None:
		return f"{int(match.group(1))} years"

	match = re.match(r"^(\d+)\s+years?$", ret)

	if match is not None:
		age = int(match.group(1))
		return f"{age} year" if age == 1 else f"{age} years"

	return pd.NA


def normalise_projection_geography(series: pd.Series) -> pd.Series:
	#### Normalise CSO projection geography labels.
	ret = series.astype("string").str.strip().replace(NUTS3_LABEL_MAPPING)

	return ret


def assign_projection_geography_type(geography: pd.Series) -> pd.Series:
	#### Assign national or NUTS3 geography type to projection rows.
	ret = geography.astype("string").str.strip().map(
		lambda value: NATIONAL_GEOGRAPHY_TYPE if value == IRELAND_GEOGRAPHY_LABEL else NUTS3_GEOGRAPHY_TYPE
	)

	return ret


def convert_projection_value_to_population(df: pd.DataFrame,
										   value_col: str = "VALUE",
										   unit_col: str = "UNIT",
										   output_col: str = "projected_population",
										   ) -> pd.DataFrame:
	#### Convert CSO projection values to persons using the reported unit.
	ret = coerce_numeric_columns(df=df,
								 columns=[value_col],
								 name="CSO projection values",
								 allow_missing=False,)

	unit_clean = ret[unit_col].astype("string").str.strip().str.lower()

	ret[output_col] = ret[value_col]
	ret.loc[unit_clean.eq("thousand"), output_col] = ret.loc[unit_clean.eq("thousand"), output_col] * 1000

	###DROP UNIT AND VALUE COLUMNS
	ret = ret.drop(columns=[unit_col, value_col], errors='ignore')


	return ret


def clean_pec26_population_projection(df: pd.DataFrame,
									  age_group_schema: dict,
									  ) -> pd.DataFrame:
	#### Clean PEC26 population projection rows and map ages to schema age groups.
	require_columns(df=df,
					required=PEC26_REQUIRED_COLUMNS,
					name="PEC26 raw projection input",)

	age_group_map = build_single_age_to_schema_group_map(age_group_schema)
	###CONVERT VALUES TO 1000 (UNIT MEASURE), drop unit and value columns
	ret = convert_projection_value_to_population(df)

	ret["projection_source"] = "PEC26"
	ret["projection_statistic"] = ret["Statistic Label"].astype("string").str.strip()
	ret["geography_name"] = normalise_projection_geography(ret["NUTS 3 Region"])
	ret["geography_type"] = assign_projection_geography_type(ret["geography_name"])
	ret["nuts3_names"] = ret["geography_name"].where(ret["geography_type"].eq(NUTS3_GEOGRAPHY_TYPE), pd.NA)
	ret["year"] = pd.to_numeric(ret["Year"], errors="coerce").astype("Int64")
	ret["sex"] = ret["Sex"].astype("string").str.strip()
	ret["projection_method"] = normalise_projection_method(ret["Criteria for Projection"])
	ret["projection_age_label_raw"] = ret["Age"].astype("string").str.strip()
	ret["projection_age_label"] = ret["projection_age_label_raw"].map(normalise_projection_age_label)
	ret["projection_age"] = ret["projection_age_label"].map(parse_single_year_age).astype("Int64")
	ret["age_group"] = ret["projection_age_label"].map(age_group_map)
	ret["is_all_ages"] = ret["projection_age_label"].eq(get_all_ages_label(age_group_schema))

	ret = ret[["projection_source",
			   "projection_statistic",
			   "geography_type",
			   "geography_name",
			   "nuts3_names",
			   "year",
			   "projection_method",
			   "sex",
			   "projection_age_label",
			   "projection_age",
			   "age_group",
			   "is_all_ages",
			   "projected_population",]].copy()

	validate_clean_pec26_population_projection(ret)

	return ret.sort_values(["geography_type",
							"geography_name",
							"projection_method",
							"sex",
							"year",
							"projection_age_label"]).reset_index(drop=True)


def validate_clean_pec26_population_projection(df: pd.DataFrame) -> None:
	#### Validate cleaned PEC26 population projection rows.
	validate_no_missing(df=df,
						columns=["projection_source",
								 "projection_statistic",
								 "geography_type",
								 "geography_name",
								 "year",
								 "projection_method",
								 "sex",
								 "projection_age_label",
								 "age_group",
								 "is_all_ages",
								 "projected_population",],
						name="Cleaned PEC26 population projection",)

	validate_non_negative(df=df,
						  columns=["projected_population"],
						  name="Cleaned PEC26 population projection",)

	validate_unique_rows(df=df,
						 keys=["geography_type",
							   "geography_name",
							   "year",
							   "projection_method",
							   "sex",
							   "projection_age_label",],
						 name="Cleaned PEC26 population projection",)


def build_projection_age_group_summary(df: pd.DataFrame,
									   age_group_schema: dict,
									   ) -> pd.DataFrame:
	#### Summarise cleaned CSO projection rows from single-year age to controlled age groups.
	all_ages_label = get_all_ages_label(age_group_schema)

	group_cols = ["projection_source",
				  "projection_statistic",
				  "geography_type",
				  "geography_name",
				  "nuts3_names",
				  "year",
				  "projection_method",
				  "sex",
				  "age_group",]

	age_group_summary = (df.loc[~df["is_all_ages"]]
		.groupby(group_cols, as_index=False, dropna=False, observed=True)
		.agg(projected_population=("projected_population", "sum"),
			 n_projection_age_labels=("projection_age_label", "nunique"),))

	all_ages_summary = (df.loc[df["is_all_ages"]]
		.assign(age_group=all_ages_label)
		.groupby(group_cols, as_index=False, dropna=False, observed=True)
		.agg(projected_population=("projected_population", "sum"),
			 n_projection_age_labels=("projection_age_label", "nunique"),))

	ret = pd.concat([age_group_summary,
					 all_ages_summary,],
					ignore_index=True)

	ret["projection_unit"] = "Number"
	ret["age_group_source"] = "config/schemas/cso_age_groups_schema.json"

	validate_projection_age_group_summary(ret)

	return ret.sort_values(["geography_type",
							"geography_name",
							"projection_method",
							"sex",
							"year",
							"age_group"]).reset_index(drop=True)


def validate_projection_age_group_summary(df: pd.DataFrame) -> None:
	#### Validate PEC26 projection age-group summary rows.
	validate_non_empty(df=df,
					   name="PEC26 population projection age-group summary",)

	validate_no_missing(df=df,
						columns=["projection_source",
								 "projection_statistic",
								 "geography_type",
								 "geography_name",
								 "year",
								 "projection_method",
								 "sex",
								 "age_group",
								 "projected_population",
								 "projection_unit",
								 "age_group_source",],
						name="PEC26 population projection age-group summary",)

	validate_non_negative(df=df,
						  columns=["projected_population",
								   "n_projection_age_labels",],
						  name="PEC26 population projection age-group summary",)

	validate_unique_rows(df=df,
						 keys=["geography_type",
							   "geography_name",
							   "year",
							   "projection_method",
							   "sex",
							   "age_group",],
						 name="PEC26 population projection age-group summary",)


def clean_pec28_dependency_ratios(df: pd.DataFrame) -> pd.DataFrame:
	#### Clean PEC28 projected dependency ratios.
	require_columns(df=df,
					required=PEC28_REQUIRED_COLUMNS,
					name="PEC28 raw dependency-ratio input",)

	###ENSURE VALUE IS NUMERIC
	ret = coerce_numeric_columns(df=df,
								 columns=["VALUE"],
								 name="PEC28 raw dependency-ratio values",
								 allow_missing=False,)

	ret["projection_source"] = "PEC28"
	ret["projection_statistic"] = ret["Statistic Label"].astype("string").str.strip()
	ret["geography_name"] = normalise_projection_geography(ret["NUTS 3 Region"])
	ret["geography_type"] = assign_projection_geography_type(ret["geography_name"])
	ret["nuts3_names"] = ret["geography_name"].where(ret["geography_type"].eq(NUTS3_GEOGRAPHY_TYPE), pd.NA)
	ret["year"] = pd.to_numeric(ret["Year"], errors="coerce").astype("Int64")
	ret["projection_method"] = normalise_projection_method(ret["Criteria for Projection"])
	ret["dependency_status"] = ret["Dependents Status"].astype("string").str.strip()
	ret["dependency_ratio_percent"] = ret["VALUE"]

	####DROP THE UNIT COLUMN
	ret = ret.drop(columns=["UNIT"], errors='ignore')

	ret = ret[["projection_source",
			   "projection_statistic",
			   "geography_type",
			   "geography_name",
			   "nuts3_names",
			   "year",
			   "projection_method",
			   "dependency_status",
			   "dependency_ratio_percent",]].copy()

	validate_clean_pec28_dependency_ratios(ret)

	return ret.sort_values(["geography_type",
							"geography_name",
							"projection_method",
							"year",
							"dependency_status"]).reset_index(drop=True)


def validate_clean_pec28_dependency_ratios(df: pd.DataFrame) -> None:
	#### Validate cleaned PEC28 dependency-ratio rows.
	validate_no_missing(df=df,
						columns=["projection_source",
								 "projection_statistic",
								 "geography_type",
								 "geography_name",
								 "year",
								 "projection_method",
								 "dependency_status",
								 "dependency_ratio_percent",],
						name="Cleaned PEC28 dependency ratios",)

	validate_non_negative(df=df,
						  columns=["dependency_ratio_percent"],
						  name="Cleaned PEC28 dependency ratios",)

	validate_unique_rows(df=df,
						 keys=["geography_type",
							   "geography_name",
							   "year",
							   "projection_method",
							   "dependency_status",],
						 name="Cleaned PEC28 dependency ratios",)


def validate_observed_county_growth_audit(df: pd.DataFrame) -> None:
	#### Validate detailed observed county growth audit rows.
	validate_non_empty(df=df,
	                   name="Observed county growth audit", )

	validate_no_missing(df=df,
	                    columns=["year",
	                             "age_group",
	                             "sex",
	                             "nuts3_names",
	                             "county_english",
	                             "county_population",
	                             "nuts3_population",
	                             "county_share_within_nuts3",
	                             "audit_source",
	                             "audit_population_source_filter",
	                             "audit_note", ],
	                    name="Observed county growth audit", )

	validate_non_negative(df=df,
	                      columns=["county_population",
	                               "nuts3_population",
	                               "county_share_within_nuts3", ],
	                      name="Observed county growth audit", )

	validate_unique_rows(df=df,
	                     keys=["year",
	                           "age_group",
	                           "sex",
	                           "county_english", ],
	                     name="Observed county growth audit", )


def validate_observed_county_share_change_summary(df: pd.DataFrame) -> None:
	#### Validate observed county share-change summary rows.
	validate_non_empty(df=df,
	                   name="Observed county share-change summary", )

	validate_no_missing(df=df,
	                    columns=["nuts3_names",
	                             "county_english",
	                             "sex",
	                             "age_group",
	                             "baseline_year",
	                             "latest_year",
	                             "n_observed_years",
	                             "baseline_county_population",
	                             "latest_county_population",
	                             "baseline_county_share_within_nuts3",
	                             "latest_county_share_within_nuts3",
	                             "audit_note", ],
	                    name="Observed county share-change summary", )

	validate_non_negative(df=df,
	                      columns=["baseline_county_population",
	                               "latest_county_population",
	                               "baseline_county_share_within_nuts3",
	                               "latest_county_share_within_nuts3", ],
	                      name="Observed county share-change summary", )

	validate_unique_rows(df=df,
	                     keys=["nuts3_names",
	                           "county_english",
	                           "sex",
	                           "age_group", ],
	                     name="Observed county share-change summary", )


def validate_observed_county_share_change_all_ages_summary(df: pd.DataFrame) -> None:
	#### Validate All-ages county share-change projection summary rows.
	validate_non_empty(df=df,
	                   name="Observed All-ages county share-change projection summary", )

	validate_no_missing(df=df,
	                    columns=["nuts3_names",
	                             "county_english",
	                             "sex",
	                             "baseline_year",
	                             "latest_year",
	                             "n_observed_years",
	                             "baseline_county_population",
	                             "latest_county_population",
	                             "baseline_county_share_within_nuts3",
	                             "latest_county_share_within_nuts3",
	                             "absolute_county_share_change",
	                             "percent_county_share_change",
	                             "static_share_assumption_flag",
	                             "projection_allocation_interpretation",
	                             "audit_note", ],
	                    name="Observed All-ages county share-change projection summary", )

	validate_non_negative(df=df,
	                      columns=["baseline_county_population",
	                               "latest_county_population",
	                               "baseline_county_share_within_nuts3",
	                               "latest_county_share_within_nuts3",
	                               "baseline_county_share_within_nuts3_percent",
	                               "latest_county_share_within_nuts3_percent", ],
	                      name="Observed All-ages county share-change projection summary", )

	validate_unique_rows(df=df,
	                     keys=["nuts3_names",
	                           "county_english",
	                           "sex", ],
	                     name="Observed All-ages county share-change projection summary", )


def prepare_observed_county_population_for_growth_audit(df: pd.DataFrame) -> pd.DataFrame:
	#### Prepare observed PEA08 county population rows for projection-allocation assumption audits.
	#### The audit uses observed county rows only where possible.
	#### This avoids treating PEA04-derived county estimates as evidence of county-level growth.
	ret = df.copy()

	ret["year"] = pd.to_numeric(ret["year"], errors="coerce").astype("Int64")
	ret["county_population"] = pd.to_numeric(ret["county_population"], errors="coerce")
	ret["age_group"] = ret["age_group"].astype("string").str.strip()
	ret["sex"] = ret["sex"].astype("string").str.strip()
	ret["nuts3_names"] = ret["nuts3_names"].astype("string").str.strip().replace(NUTS3_LABEL_MAPPING)
	ret["county_english"] = ret["county_english"].astype("string").str.strip()
	ret["population_source"] = ret["population_source"].astype("string").str.strip()

	if "is_interpolated" in ret.columns:
		ret["is_interpolated"] = ret["is_interpolated"].astype("string").str.lower().isin(["true", "1", "yes"])
	else:
		ret["is_interpolated"] = False

	ret = ret.loc[ret["population_source"].eq(OBSERVED_COUNTY_POPULATION_SOURCE)].copy()
	ret = ret.loc[~ret["is_interpolated"]].copy()

	validate_non_empty(df=ret,
	                   name="Observed PEA08 county population rows for projection allocation audit", )

	validate_no_missing(df=ret,
	                    columns=["year",
	                             "age_group",
	                             "sex",
	                             "nuts3_names",
	                             "county_english",
	                             "county_population",
	                             "population_source",
	                             "is_interpolated", ],
	                    name="Observed PEA08 county population rows for projection allocation audit", )

	validate_non_negative(df=ret,
	                      columns=["county_population"],
	                      name="Observed PEA08 county population rows for projection allocation audit", )

	validate_unique_rows(df=ret,
	                     keys=["year",
	                           "age_group",
	                           "sex",
	                           "county_english", ],
	                     name="Observed PEA08 county population rows for projection allocation audit", )

	return ret


def build_observed_county_growth_audit(county_population: pd.DataFrame) -> pd.DataFrame:
	#### Build annual observed county growth audit for future projection-allocation decisions.
	#### This table describes historical county growth patterns.
	#### It should not be interpreted as a future county projection.
	#### It is intended to show whether static NUTS3-to-county shares are likely to hide urban/rural growth differences.
	ret = prepare_observed_county_population_for_growth_audit(county_population)

	nuts3_totals = (ret.groupby(["year",
	                             "age_group",
	                             "sex",
	                             "nuts3_names", ],
	                            as_index=False,
	                            dropna=False,
	                            observed=True, )
	                .agg(nuts3_population=("county_population", "sum"), ))

	ret = ret.merge(nuts3_totals,
	                on=["year",
	                    "age_group",
	                    "sex",
	                    "nuts3_names", ],
	                how="left",
	                validate="many_to_one", )

	ret["county_share_within_nuts3"] = ret["county_population"] / ret["nuts3_population"]

	ret = ret.sort_values(["nuts3_names",
	                       "county_english",
	                       "sex",
	                       "age_group",
	                       "year", ]).reset_index(drop=True)

	group_cols = ["nuts3_names",
	              "county_english",
	              "sex",
	              "age_group", ]

	grouped = ret.groupby(group_cols, observed=True)

	ret["previous_year"] = grouped["year"].shift(1)
	ret["previous_county_population"] = grouped["county_population"].shift(1)
	ret["previous_county_share_within_nuts3"] = grouped["county_share_within_nuts3"].shift(1)

	ret["absolute_county_population_change"] = ret["county_population"] - ret["previous_county_population"]

	ret["percent_county_population_change"] = (
			ret["absolute_county_population_change"]
			/ ret["previous_county_population"]
			* 100
	)

	ret["annual_growth_factor"] = ret["county_population"] / ret["previous_county_population"]

	ret["absolute_share_change"] = (
			ret["county_share_within_nuts3"]
			- ret["previous_county_share_within_nuts3"]
	)

	ret["percent_share_change"] = (
			ret["absolute_share_change"]
			/ ret["previous_county_share_within_nuts3"]
			* 100
	)

	ret["audit_source"] = "dataset/processed/preprocessing/cso_county_population_combined_pea04_pea08.csv"
	ret["audit_population_source_filter"] = OBSERVED_COUNTY_POPULATION_SOURCE
	ret["audit_note"] = (
		"Observed county growth audit for projection allocation assumptions. "
		"This output does not create future county projections. "
		"It is used to assess whether static NUTS3-to-county shares are defensible before any HSE-region projection allocation is implemented."
	)

	validate_observed_county_growth_audit(ret)

	return ret


def build_observed_county_share_change_summary(county_growth_audit: pd.DataFrame) -> pd.DataFrame:
	#### Summarise observed county share changes within NUTS3 areas.
	#### This summary supports later decisions about static versus trend-adjusted NUTS3-to-county allocation.
	records = []

	group_cols = ["nuts3_names",
	              "county_english",
	              "sex",
	              "age_group", ]

	for group_values, group_df in county_growth_audit.groupby(group_cols, observed=True):
		group_df = group_df.sort_values("year").reset_index(drop=True)

		if group_df.empty:
			continue

		first_row = group_df.iloc[0]
		last_row = group_df.iloc[-1]

		baseline_population = first_row["county_population"]
		latest_population = last_row["county_population"]
		baseline_share = first_row["county_share_within_nuts3"]
		latest_share = last_row["county_share_within_nuts3"]

		record = dict(zip(group_cols, group_values))

		record["baseline_year"] = int(first_row["year"])
		record["latest_year"] = int(last_row["year"])
		record["n_observed_years"] = int(group_df["year"].nunique())

		record["baseline_county_population"] = baseline_population
		record["latest_county_population"] = latest_population
		record["absolute_county_population_change"] = latest_population - baseline_population

		if pd.notna(baseline_population) and baseline_population != 0:
			record["percent_county_population_change"] = (
					(latest_population - baseline_population)
					/ baseline_population
					* 100
			)
		else:
			record["percent_county_population_change"] = pd.NA

		record["baseline_county_share_within_nuts3"] = baseline_share
		record["latest_county_share_within_nuts3"] = latest_share
		record["absolute_county_share_change"] = latest_share - baseline_share

		if pd.notna(baseline_share) and baseline_share != 0:
			record["percent_county_share_change"] = (
					(latest_share - baseline_share)
					/ baseline_share
					* 100
			)
		else:
			record["percent_county_share_change"] = pd.NA

		record["mean_annual_growth_factor"] = group_df["annual_growth_factor"].mean(skipna=True)
		record["mean_annual_percent_population_change"] = group_df["percent_county_population_change"].mean(skipna=True)
		record["mean_annual_share_change"] = group_df["absolute_share_change"].mean(skipna=True)

		record["audit_note"] = (
			"Observed county share-change summary for projection allocation assumptions. "
			"This table summarises historical county growth and within-NUTS3 share movement. "
			"It should inform, but not itself define, any future county or HSE projection allocation scenario."
		)

		records.append(record)

	ret = pd.DataFrame.from_records(records)

	validate_observed_county_share_change_summary(ret)

	return ret.sort_values(["nuts3_names",
	                        "county_english",
	                        "sex",
	                        "age_group", ]).reset_index(drop=True)


def classify_static_share_assumption_flag(absolute_share_change: float | pd.NA) -> str | pd.NA:
	#### Classify observed county share movement to support projection allocation assumptions.
	if pd.isna(absolute_share_change):
		return pd.NA

	absolute_change = abs(absolute_share_change)

	if absolute_change < LOW_SHARE_CHANGE_THRESHOLD:
		return "low_observed_share_change"

	if absolute_change < HIGH_SHARE_CHANGE_THRESHOLD:
		return "moderate_observed_share_change"

	return "high_observed_share_change"


def build_observed_county_share_change_all_ages_summary(county_share_change: pd.DataFrame) -> pd.DataFrame:
	#### Build All-ages county share-change summary for projection allocation decisions.
	#### This table is a compact checkpoint for assessing whether static NUTS3-to-county shares are defensible.
	#### It does not create projected county populations or HSE-region projections.
	ret = county_share_change.copy()

	ret = ret.loc[ret["age_group"].eq("All ages")].copy()

	validate_non_empty(df=ret,
	                   name="Observed All-ages county share-change projection summary input", )

	keep_cols = ["nuts3_names",
	             "county_english",
	             "sex",
	             "baseline_year",
	             "latest_year",
	             "n_observed_years",
	             "baseline_county_population",
	             "latest_county_population",
	             "absolute_county_population_change",
	             "percent_county_population_change",
	             "baseline_county_share_within_nuts3",
	             "latest_county_share_within_nuts3",
	             "absolute_county_share_change",
	             "percent_county_share_change",
	             "mean_annual_growth_factor",
	             "mean_annual_percent_population_change",
	             "mean_annual_share_change", ]

	require_columns(df=ret,
	                required=set(keep_cols),
	                name="Observed All-ages county share-change projection summary input", )

	ret = ret[keep_cols].copy()

	ret["absolute_county_share_change_percentage_points"] = ret["absolute_county_share_change"] * 100
	ret["baseline_county_share_within_nuts3_percent"] = ret["baseline_county_share_within_nuts3"] * 100
	ret["latest_county_share_within_nuts3_percent"] = ret["latest_county_share_within_nuts3"] * 100

	ret["static_share_assumption_flag"] = ret["absolute_county_share_change"].map(
		classify_static_share_assumption_flag
	)

	ret["projection_allocation_interpretation"] = ret["static_share_assumption_flag"].map(
		{
			"low_observed_share_change": "Observed county share changed by less than 1 percentage point between baseline and latest year.",
			"moderate_observed_share_change": "Observed county share changed by 1 to less than 3 percentage points between baseline and latest year.",
			"high_observed_share_change": "Observed county share changed by 3 or more percentage points between baseline and latest year.", }
	)

	ret["audit_note"] = (
		"All-ages observed county share-change summary for projection allocation assumptions. "
		"This table should be used to assess whether static NUTS3-to-county shares are acceptable "
		"before deriving any HSE-region projection scenario from PEC26 NUTS3 projections."
	)

	validate_observed_county_share_change_all_ages_summary(ret)

	return ret.sort_values(["static_share_assumption_flag",
	                        "nuts3_names",
	                        "county_english",
	                        "sex", ]).reset_index(drop=True)


def write_projection_output(df: pd.DataFrame,
							output_path: Path,
							manifest_dir: Path,
							stage: str,
							) -> Path:
	#### Write one cleaned CSO projection output with audit and manifest files.
	ret = save_dataset_with_manifest(df=df,
									 output_path=output_path,
									 manifest_dir=manifest_dir,
									 stage=stage,
									 write_audit=True,
									 write_json_manifest=True,
									 index=False,)

	return ret


def write_projection_outputs(pec26_age: pd.DataFrame,
							 pec26_age_group: pd.DataFrame,
							 pec28_dependency: pd.DataFrame,
							 county_growth_audit: pd.DataFrame,
							 county_share_change_audit: pd.DataFrame,
							 county_share_change_all_ages_summary: pd.DataFrame,
							 dirs: dict[str, Path],
							 ) -> dict[str, Path]:
	#### Write all cleaned CSO projection outputs.
	outputs = {"pec26_age": dirs["processed"] / PEC26_AGE_OUTPUT_FILE,
	           "pec26_age_group": dirs["processed"] / PEC26_AGE_GROUP_OUTPUT_FILE,
	           "pec28_dependency": dirs["processed"] / PEC28_DEPENDENCY_OUTPUT_FILE,
	           "county_growth_audit": dirs["processed"] / COUNTY_GROWTH_AUDIT_OUTPUT_FILE,
	           "county_share_change_audit": dirs["processed"] / COUNTY_SHARE_CHANGE_AUDIT_OUTPUT_FILE,
			   "county_share_change_all_ages_summary": dirs["processed"] / COUNTY_SHARE_CHANGE_ALL_AGES_SUMMARY_OUTPUT_FILE,}

	write_projection_output(df=pec26_age,
							output_path=outputs["pec26_age"],
							manifest_dir=dirs["manifests"],
							stage="cso_pec26_population_projection_age",)

	write_projection_output(df=pec26_age_group,
							output_path=outputs["pec26_age_group"],
							manifest_dir=dirs["manifests"],
							stage="cso_pec26_population_projection_age_group",)

	write_projection_output(df=pec28_dependency,
							output_path=outputs["pec28_dependency"],
							manifest_dir=dirs["manifests"],
							stage="cso_pec28_dependency_ratios",)

	write_projection_output(df=county_growth_audit,
	                        output_path=outputs["county_growth_audit"],
	                        manifest_dir=dirs["manifests"],
	                        stage="cso_observed_county_growth_for_projection_audit", )

	write_projection_output(df=county_share_change_audit,
	                        output_path=outputs["county_share_change_audit"],
	                        manifest_dir=dirs["manifests"],
	                        stage="cso_observed_county_share_change_for_projection_audit", )

	write_projection_output(df=county_share_change_all_ages_summary,
	                        output_path=outputs["county_share_change_all_ages_summary"],
	                        manifest_dir=dirs["manifests"],
	                        stage="cso_observed_county_share_change_all_ages_projection_summary", )

	return outputs


def run_cso_projection_preprocessing(paths: ProjectPaths | None = None) -> dict[str, Path]:
	#### Run CSO projection preprocessing for manually downloaded PEC datasets.
	if paths is None:
		paths = get_project_paths()

	stage = "preprocessing"
	dirs = paths.stage_dirs(stage)

	logger = get_logger(name="preprocess_cso_projections",
						log_dir=dirs["logs"],
						filename="preprocess_cso_projections.log",)

	logger.info("Starting CSO projection preprocessing")

	age_group_schema = load_cso_age_group_schema(paths)
	raw = load_cso_projection_raw_files(paths)
	observed_county_population = load_observed_county_population_for_projection_audit(dirs)

	write_column_audit(raw["pec26"],
					   dirs["manifests"] / "cso_pec26_raw_column_audit.csv",)
	write_column_audit(raw["pec28"],
					   dirs["manifests"] / "cso_pec28_raw_column_audit.csv",)

	pec26_age = clean_pec26_population_projection(df=raw["pec26"],
												  age_group_schema=age_group_schema,)
	pec26_age_group = build_projection_age_group_summary(df=pec26_age,
														 age_group_schema=age_group_schema,)
	pec28_dependency = clean_pec28_dependency_ratios(raw["pec28"])
	county_growth_audit = build_observed_county_growth_audit(observed_county_population)
	county_share_change_audit = build_observed_county_share_change_summary(county_growth_audit)

	county_share_change_all_ages_summary = build_observed_county_share_change_all_ages_summary(
		county_share_change_audit
	)

	outputs = write_projection_outputs(pec26_age=pec26_age,
	                                   pec26_age_group=pec26_age_group,
	                                   pec28_dependency=pec28_dependency,
	                                   county_growth_audit=county_growth_audit,
	                                   county_share_change_audit=county_share_change_audit,
	                                   county_share_change_all_ages_summary=county_share_change_all_ages_summary,
	                                   dirs=dirs, )

	log_block(logger,
			  "CSO projection preprocessing completed",
			  {"pec26_age_rows": int(pec26_age.shape[0]),
			   "pec26_age_group_rows": int(pec26_age_group.shape[0]),
			   "pec28_dependency_rows": int(pec28_dependency.shape[0]),
			   "pec26_year_min": int(pec26_age["year"].min()),
			   "pec26_year_max": int(pec26_age["year"].max()),
			   "pec28_year_min": int(pec28_dependency["year"].min()),
			   "pec28_year_max": int(pec28_dependency["year"].max()),
			   "pec26_geography_types": sorted(pec26_age["geography_type"].dropna().unique().tolist()),
			   "pec28_geography_types": sorted(pec28_dependency["geography_type"].dropna().unique().tolist()),
			   "county_growth_audit_rows": int(county_growth_audit.shape[0]),
			   "county_share_change_audit_rows": int(county_share_change_audit.shape[0]),
			   "county_growth_audit_year_min": int(county_growth_audit["year"].min()),
			   "county_growth_audit_year_max": int(county_growth_audit["year"].max()),
			   "county_share_change_all_ages_summary_rows": int(county_share_change_all_ages_summary.shape[0]),
			   "county_share_change_all_ages_high_flags": int(county_share_change_all_ages_summary["static_share_assumption_flag"]
															  .eq("high_observed_share_change").sum()),
			   "note": ("PEC26 is treated as the primary population projection source because it includes both Ireland "
						"and NUTS3 rows. PEC19 is not used in this first projection baseline. "
						"PEC26 NUTS3 rows are not allocated to HSE regions in this preprocessing step. "
						"Observed PEA08 county growth audits are written as companion outputs to assess whether future "
						"NUTS3-to-county allocation assumptions should use static or trend-adjusted shares."),},)

	logger.info("Finished CSO projection preprocessing")

	return outputs


def main() -> None:
	#### Run CSO projection preprocessing from the command line.
	outputs = run_cso_projection_preprocessing()

	print("CSO projection preprocessing completed.")
	print(outputs)


if __name__ == "__main__":
	main()