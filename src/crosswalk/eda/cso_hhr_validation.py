#### cso_hhr_validation.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crosswalk.eda.common import read_processed_csv, write_eda_table
from crosswalk.paths import ProjectPaths, get_project_paths
from crosswalk.preprocessing.preprocess_cso_projections import (
	OBSERVED_COUNTY_POPULATION_SOURCE,
	find_latest_raw_file,
)
from crosswalk.preprocessing.reference import (
	harmonize_hse_region_series,
	order_hse_region_rows,
)
from crosswalk.utils.io import ensure_directories, read_required_csv
from crosswalk.utils.logging import get_logger, log_block
from crosswalk.utils.validation import (
	require_columns,
	validate_non_empty,
	validate_non_negative,
	validate_no_missing,
)


CSO_MASTER_FILE = "cso_population_master_age_sex_region_county_year.csv"
GEO_HSE_REGION_SEX_SHARES_FILE = "geo_census_2022_hse_region_sex_shares.csv"
HHR01_RAW_PATTERN = "HHR01*.csv"
HHR02_RAW_PATTERN = "HHR02*.csv"
HHR01_COUNT_STATISTIC = "HHR01C01"
HHR02_SHARE_STATISTIC = "HHR02C01"
VALIDATION_YEAR = 2022
HHR_IMPLIED_OUTPUT_FILE = "cso_hhr_implied_hse_age_sex_2022_counts.csv"
HHR_WEATHERED_VALIDATION_OUTPUT_FILE = "cso_hhr_weathered_hse_age_sex_2022_validation.csv"
HHR_VALIDATION_SUMMARY_OUTPUT_FILE = "cso_hhr_validation_summary.csv"
LEGACY_HHR_OUTPUT_FILES = (
	"cso_hhr01_hse_population_2022_validation.csv",
	"cso_hhr02_hse_age_sex_2022_validation.csv",
)
REGIONAL_TOTAL_TOLERANCE = 0.001
RELATIVE_ERROR_PASS_THRESHOLD = 0.01
RELATIVE_ERROR_WARN_THRESHOLD = 0.05


def load_hhr_validation_inputs(paths: ProjectPaths) -> dict[str, pd.DataFrame]:
	#### Load WEATHERED master population, geo sex shares, and raw CSO HHR reference tables.
	hhr01_path = find_latest_raw_file(paths=paths,
									  pattern=HHR01_RAW_PATTERN,)
	hhr02_path = find_latest_raw_file(paths=paths,
									  pattern=HHR02_RAW_PATTERN,)

	ret = {"cso_master": read_processed_csv(paths=paths,
											 filename=CSO_MASTER_FILE,),
		   "hse_region_sex_shares": read_processed_csv(paths=paths,
													   filename=GEO_HSE_REGION_SEX_SHARES_FILE,),
		   "hhr01_raw": read_required_csv(path=hhr01_path,
										  name=f"CSO HHR01 reference file {hhr01_path.name}",),
		   "hhr02_raw": read_required_csv(path=hhr02_path,
										  name=f"CSO HHR02 reference file {hhr02_path.name}",),}

	return ret


def clean_hse_region_sex_shares(df: pd.DataFrame,
								hhr01_reference: pd.DataFrame,
								validation_year: int = VALIDATION_YEAR,
								tolerance: float = REGIONAL_TOTAL_TOLERANCE,
								) -> pd.DataFrame:
	#### Clean Census 2022 HSE-region sex shares and validate against HHR01 regional totals.
	ret = df.copy()

	require_columns(df=ret,
					required={"year",
							  "hse_region",
							  "census_male",
							  "census_female",
							  "census_total",
							  "male_share",
							  "female_share"},
					name="HSE-region sex shares for HHR validation",)

	ret["year"] = pd.to_numeric(ret["year"], errors="coerce").astype("Int64")
	ret["hse_region"] = harmonize_hse_region_series(ret["hse_region"].astype("string").str.strip())

	for column in ["census_male", "census_female", "census_total", "male_share", "female_share"]:
		ret[column] = pd.to_numeric(ret[column], errors="coerce")

	ret = ret.loc[ret["year"].eq(validation_year)].copy()
	ret = ret[["year",
			   "hse_region",
			   "census_male",
			   "census_female",
			   "census_total",
			   "male_share",
			   "female_share"]].drop_duplicates()

	validate_non_empty(df=ret,
					   name=f"HSE-region sex shares for year {validation_year}",)

	validate_no_missing(df=ret,
						columns=["year",
								 "hse_region",
								 "census_male",
								 "census_female",
								 "census_total",
								 "male_share",
								 "female_share"],
						name="HSE-region sex shares",)

	share_check = (ret["male_share"] + ret["female_share"] - 1).abs()

	if share_check.gt(0.0001).any():
		examples = ret.loc[share_check.gt(0.0001), ["hse_region", "male_share", "female_share"]].head(10)
		raise ValueError(f"HSE-region sex shares do not sum to 1: {examples.to_dict(orient='records')}")

	sex_total_check = (ret["census_male"] + ret["census_female"] - ret["census_total"]).abs()

	if sex_total_check.gt(0).any():
		examples = ret.loc[sex_total_check.gt(0), ["hse_region", "census_male", "census_female", "census_total"]].head(10)
		raise ValueError(f"HSE-region male plus female does not equal total: {examples.to_dict(orient='records')}")

	alignment = ret.merge(hhr01_reference,
						  on=["year", "hse_region"],
						  how="inner",
						  validate="one_to_one",)

	if alignment.shape[0] != ret.shape[0]:
		missing = sorted(set(ret["hse_region"].astype(str)) - set(alignment["hse_region"].astype(str)))
		raise ValueError(f"HSE-region sex shares missing HHR01 regional totals for: {missing}")

	alignment["regional_total_gap"] = (
		(alignment["census_total"] - alignment["hhr01_regional_total"]).abs()
		/ alignment["hhr01_regional_total"]
	)

	if alignment["regional_total_gap"].gt(tolerance).any():
		examples = alignment.loc[alignment["regional_total_gap"].gt(tolerance),
								 ["hse_region", "census_total", "hhr01_regional_total", "regional_total_gap"]].head(10)
		raise ValueError(
			f"HSE-region Census sex totals differ from HHR01 regional totals: {examples.to_dict(orient='records')}"
		)

	ret["census_national_total"] = float(ret["census_total"].sum())

	return order_hse_region_rows(ret.sort_values(["year", "hse_region"]).reset_index(drop=True))


def clean_hhr01_reference(df: pd.DataFrame,
						  validation_year: int = VALIDATION_YEAR,
						  ) -> pd.DataFrame:
	#### Clean HHR01 Census HSE-region population counts.
	ret = df.copy()

	ret = ret.rename(columns={"Year": "year",
							  "HSE Health Regions": "hse_region",
							  "VALUE": "hhr01_regional_total",})

	if "hse_region" not in ret.columns and "HSE Regions" in ret.columns:
		ret = ret.rename(columns={"HSE Regions": "hse_region"})

	require_columns(df=ret,
					required={"STATISTIC", "year", "hse_region", "hhr01_regional_total"},
					name="HHR01 reference after rename",)

	ret["year"] = pd.to_numeric(ret["year"], errors="coerce").astype("Int64")
	ret["hhr01_regional_total"] = pd.to_numeric(ret["hhr01_regional_total"], errors="coerce")
	ret["hse_region"] = harmonize_hse_region_series(ret["hse_region"].astype("string").str.strip())

	ret = ret.loc[ret["STATISTIC"].eq(HHR01_COUNT_STATISTIC)
				  & ret["year"].eq(validation_year)].copy()

	ret = ret[["year", "hse_region", "hhr01_regional_total"]].drop_duplicates()

	validate_non_empty(df=ret,
					   name=f"HHR01 cleaned reference rows for year {validation_year}",)

	validate_no_missing(df=ret,
						columns=["year", "hse_region", "hhr01_regional_total"],
						name="HHR01 cleaned reference",)

	validate_non_negative(df=ret,
						  columns=["hhr01_regional_total"],
						  name="HHR01 cleaned reference",)

	return order_hse_region_rows(ret.sort_values(["year", "hse_region"]).reset_index(drop=True))


def clean_hhr02_reference(df: pd.DataFrame,
						  validation_year: int = VALIDATION_YEAR,
						  ) -> pd.DataFrame:
	#### Clean HHR02 age-sex shares within each HSE region.
	ret = df.copy()

	ret = ret.rename(columns={"Year": "year",
							  "Age Group": "age_group",
							  "Sex": "sex",
							  "VALUE": "hhr02_share_percent",})

	if "hse_region" not in ret.columns:
		for candidate in ("HSE Regions", "HSE Health Regions"):
			if candidate in ret.columns:
				ret = ret.rename(columns={candidate: "hse_region"})
				break

	require_columns(df=ret,
					required={"STATISTIC", "year", "hse_region", "sex", "age_group", "hhr02_share_percent"},
					name="HHR02 reference after rename",)

	ret["year"] = pd.to_numeric(ret["year"], errors="coerce").astype("Int64")
	ret["hhr02_share_percent"] = pd.to_numeric(ret["hhr02_share_percent"], errors="coerce")
	ret["hse_region"] = harmonize_hse_region_series(ret["hse_region"].astype("string").str.strip())
	ret["age_group"] = ret["age_group"].astype("string").str.strip()
	ret["sex"] = ret["sex"].astype("string").str.strip()

	ret = ret.loc[ret["STATISTIC"].eq(HHR02_SHARE_STATISTIC)
				  & ret["year"].eq(validation_year)
				  & ret["sex"].isin(["Male", "Female"])
				  & ret["age_group"].ne("All ages")].copy()

	ret["hhr02_share_within_sex_region"] = ret["hhr02_share_percent"] / 100.0

	ret = ret[["year",
			   "hse_region",
			   "sex",
			   "age_group",
			   "hhr02_share_percent",
			   "hhr02_share_within_sex_region",]].drop_duplicates()

	validate_non_empty(df=ret,
					   name=f"HHR02 cleaned reference rows for year {validation_year}",)

	validate_no_missing(df=ret,
						columns=["year",
								 "hse_region",
								 "sex",
								 "age_group",
								 "hhr02_share_within_sex_region"],
						name="HHR02 cleaned reference",)

	share_check = (ret.groupby(["hse_region", "sex"], as_index=False, observed=True)
				   .agg(hhr02_share_sum=("hhr02_share_within_sex_region", "sum"),))

	share_check["share_sum_error"] = (share_check["hhr02_share_sum"] - 1).abs()

	if share_check["share_sum_error"].gt(0.02).any():
		examples = share_check.loc[share_check["share_sum_error"].gt(0.02)].head(10).to_dict(orient="records")
		raise ValueError(f"HHR02 age shares do not sum to 1 within region-sex strata: {examples}")

	return ret.sort_values(["hse_region", "sex", "age_group"]).reset_index(drop=True)


def build_hhr_implied_age_sex_counts(hhr01_reference: pd.DataFrame,
								   hhr02_reference: pd.DataFrame,
								   hse_region_sex_shares: pd.DataFrame,
								   ) -> pd.DataFrame:
	#### Build Census-implied HSE age-sex counts from HHR01, HHR02, and HSE-region sex shares.
	ret = hhr02_reference.merge(hhr01_reference,
								on=["year", "hse_region"],
								how="inner",
								validate="many_to_one",)

	ret = ret.merge(hse_region_sex_shares,
					on=["year", "hse_region"],
					how="inner",
					validate="many_to_one",)

	validate_non_empty(df=ret,
					   name="HHR implied age-sex count rows",)

	ret["hse_region_male_share"] = ret["male_share"]
	ret["hse_region_female_share"] = ret["female_share"]
	ret["hhr_sex_region_total"] = ret["hhr01_regional_total"] * ret["hse_region_male_share"].where(
		ret["sex"].eq("Male"),
		ret["hse_region_female_share"],
	)
	ret["hhr_implied_population"] = (
		ret["hhr_sex_region_total"]
		* ret["hhr02_share_within_sex_region"]
	)
	ret["reference_dataset"] = (
		"HHR01 + HHR02 + geo_census_2022_hse_region_sex_shares.csv"
	)
	ret["interpretation_note"] = (
		"Census-implied population count by HSE region, sex, and age group. "
		"Regional denominators come from HHR01; age shares come from HHR02 within each sex; "
		"regional male and female totals are allocated using Census 2022 HSE-region sex shares "
		"derived from Table 1.2 small-area counts."
	)

	output_columns = ["year",
					  "hse_region",
					  "sex",
					  "age_group",
					  "hhr02_share_percent",
					  "hhr02_share_within_sex_region",
					  "hhr01_regional_total",
					  "census_national_total",
					  "census_male",
					  "census_female",
					  "census_total",
					  "hse_region_male_share",
					  "hse_region_female_share",
					  "hhr_sex_region_total",
					  "hhr_implied_population",
					  "reference_dataset",
					  "interpretation_note"]

	return ret[output_columns].sort_values(["hse_region", "sex", "age_group"]).reset_index(drop=True)


def build_weathered_hse_population_reference(master: pd.DataFrame,
											 validation_year: int = VALIDATION_YEAR,
											 ) -> tuple[pd.DataFrame, pd.DataFrame]:
	#### Build WEATHERED HSE-region totals and age-sex counts for the validation year.
	ret = master.copy()

	require_columns(df=ret,
					required={"year",
							  "age_group",
							  "sex",
							  "hse_region",
							  "population",
							  "population_source",
							  "is_interpolated",},
					name="CSO population master for HHR validation",)

	ret["year"] = pd.to_numeric(ret["year"], errors="coerce").astype("Int64")
	ret["population"] = pd.to_numeric(ret["population"], errors="coerce")
	ret["is_interpolated"] = ret["is_interpolated"].astype("string").str.lower().isin(["true", "1", "yes"])
	ret["population_source"] = ret["population_source"].astype("string").str.strip()
	ret["hse_region"] = harmonize_hse_region_series(ret["hse_region"].astype("string").str.strip())
	ret["age_group"] = ret["age_group"].astype("string").str.strip()
	ret["sex"] = ret["sex"].astype("string").str.strip()

	ret = ret.loc[ret["year"].eq(validation_year)
				  & ret["population_source"].eq(OBSERVED_COUNTY_POPULATION_SOURCE)
				  & ~ret["is_interpolated"]
				  & ret["sex"].isin(["Male", "Female"])].copy()

	validate_non_empty(df=ret,
					   name=f"Observed WEATHERED HSE population rows for year {validation_year}",)

	all_ages = ret.loc[ret["age_group"].eq("All ages")].copy()

	validate_non_empty(df=all_ages,
					   name=f"Observed WEATHERED HSE All ages rows for year {validation_year}",)

	regional_totals = (all_ages.groupby(["year", "hse_region"], as_index=False, observed=True)
					   .agg(weathered_hse_population=("population", "sum"),)
					   .sort_values(["year", "hse_region"])
					   .reset_index(drop=True))

	age_sex = (ret.loc[~ret["age_group"].eq("All ages")]
			   .groupby(["year", "hse_region", "sex", "age_group"], as_index=False, observed=True)
			   .agg(weathered_hse_population=("population", "sum"),)
			   .sort_values(["hse_region", "sex", "age_group"])
			   .reset_index(drop=True))

	validate_non_empty(df=age_sex,
					   name=f"Observed WEATHERED HSE age-sex rows for year {validation_year}",)

	return regional_totals, age_sex


def _relative_error(reference: pd.Series,
					comparison: pd.Series,
					) -> pd.Series:
	#### Return signed relative error as a fraction of the reference value.
	denominator = reference.replace(0, pd.NA)

	return (comparison - reference) / denominator


def _validation_flag(relative_error: pd.Series,
					 pass_threshold: float = RELATIVE_ERROR_PASS_THRESHOLD,
					 warn_threshold: float = RELATIVE_ERROR_WARN_THRESHOLD,
					 ) -> pd.Series:
	#### Classify validation rows by absolute relative error thresholds.
	abs_error = relative_error.abs()

	return abs_error.map(lambda value: "pass"
						 if pd.notna(value) and value <= pass_threshold
						 else "warn"
						 if pd.notna(value) and value <= warn_threshold
						 else "fail")


def build_hhr_weathered_validation(hhr_implied: pd.DataFrame,
								   weathered_age_sex: pd.DataFrame,
								   ) -> pd.DataFrame:
	#### Compare WEATHERED age-sex counts with Census-implied HHR reference counts.
	ret = weathered_age_sex.merge(
		hhr_implied,
		on=["year", "hse_region", "sex", "age_group"],
		how="inner",
		validate="one_to_one",
	)

	validate_non_empty(df=ret,
					   name="HHR versus WEATHERED age-sex validation comparison",)

	ret = ret.rename(columns={"weathered_hse_population": "weathered_population"})

	ret["absolute_population_error"] = ret["weathered_population"] - ret["hhr_implied_population"]
	ret["signed_relative_error"] = _relative_error(reference=ret["hhr_implied_population"],
												   comparison=ret["weathered_population"],)
	ret["absolute_relative_error"] = ret["signed_relative_error"].abs()
	ret["validation_flag"] = _validation_flag(relative_error=ret["signed_relative_error"])
	ret["validation_type"] = "hhr_implied_hse_age_sex_count"
	ret["comparison_dataset"] = CSO_MASTER_FILE
	ret["interpretation_note"] = (
		"Compares WEATHERED 2022 observed PEA08 county populations allocated to HSE regions "
		"with Census-implied HHR01 + HHR02 age-sex counts."
	)

	return ret.sort_values(["hse_region", "sex", "age_group"]).reset_index(drop=True)


def build_hhr_regional_total_validation(hhr01_reference: pd.DataFrame,
										weathered_regional_totals: pd.DataFrame,
										) -> pd.DataFrame:
	#### Compare WEATHERED HSE-region totals with HHR01 Census 2022 reference counts.
	ret = weathered_regional_totals.merge(hhr01_reference,
										  on=["year", "hse_region"],
										  how="inner",
										  validate="one_to_one",)

	validate_non_empty(df=ret,
					   name="HHR01 regional total validation comparison",)

	ret["absolute_population_error"] = ret["weathered_hse_population"] - ret["hhr01_regional_total"]
	ret["signed_relative_error"] = _relative_error(reference=ret["hhr01_regional_total"],
												   comparison=ret["weathered_hse_population"],)
	ret["absolute_relative_error"] = ret["signed_relative_error"].abs()

	national = (ret.groupby("year", as_index=False, observed=True)
				.agg(reference_national_population=("hhr01_regional_total", "sum"),
					 weathered_national_population=("weathered_hse_population", "sum"),))

	ret = ret.merge(national,
					on="year",
					how="left",
					validate="many_to_one",)

	ret["reference_hse_population_share"] = ret["hhr01_regional_total"] / ret["reference_national_population"]
	ret["weathered_hse_population_share"] = ret["weathered_hse_population"] / ret["weathered_national_population"]
	ret["share_difference"] = ret["weathered_hse_population_share"] - ret["reference_hse_population_share"]
	ret["validation_flag"] = _validation_flag(relative_error=ret["signed_relative_error"])

	return order_hse_region_rows(ret.sort_values(["year", "hse_region"]).reset_index(drop=True))


def build_hhr_validation_summary(hhr_regional_validation: pd.DataFrame,
								 hhr_weathered_validation: pd.DataFrame,
								 ) -> pd.DataFrame:
	#### Summarise HHR regional and age-sex validation checks for the Census reference year.
	records = []

	hhr01_max = float(hhr_regional_validation["absolute_relative_error"].max())
	hhr01_failures = int(hhr_regional_validation["validation_flag"].eq("fail").sum())

	records.append({"validation_section": "hhr01_hse_region_total",
					"validation_metric": "max_absolute_relative_error",
					"validation_value": hhr01_max,
					"n_rows": int(hhr_regional_validation.shape[0]),
					"n_fail_rows": hhr01_failures,
					"pass_threshold": RELATIVE_ERROR_PASS_THRESHOLD,
					"warn_threshold": RELATIVE_ERROR_WARN_THRESHOLD,
					"validation_status": "pass"
					if hhr01_max <= RELATIVE_ERROR_PASS_THRESHOLD and hhr01_failures == 0
					else "warn"
					if hhr01_max <= RELATIVE_ERROR_WARN_THRESHOLD
					else "fail",
					"interpretation_note": (
						"HHR01 compares WEATHERED 2022 HSE-region totals with CSO Census 2022 reference totals."
					),})

	age_sex_max = float(hhr_weathered_validation["absolute_relative_error"].max())
	age_sex_failures = int(hhr_weathered_validation["validation_flag"].eq("fail").sum())

	records.append({"validation_section": "hhr_implied_hse_age_sex_count",
					"validation_metric": "max_absolute_relative_error",
					"validation_value": age_sex_max,
					"n_rows": int(hhr_weathered_validation.shape[0]),
					"n_fail_rows": age_sex_failures,
					"pass_threshold": RELATIVE_ERROR_PASS_THRESHOLD,
					"warn_threshold": RELATIVE_ERROR_WARN_THRESHOLD,
					"validation_status": "pass"
					if age_sex_max <= RELATIVE_ERROR_PASS_THRESHOLD and age_sex_failures == 0
					else "warn"
					if age_sex_max <= RELATIVE_ERROR_WARN_THRESHOLD
					else "fail",
					"interpretation_note": (
						"Compares WEATHERED 2022 HSE age-sex counts with Census-implied counts "
						"derived from HHR01 regional totals, HHR02 within-sex age shares, "
						"and Census 2022 HSE-region sex shares from geo preprocessing."
					),})

	national_reference = float(hhr_regional_validation["reference_national_population"].iloc[0])
	national_weathered = float(hhr_regional_validation["weathered_national_population"].iloc[0])
	national_error = (national_weathered - national_reference) / national_reference

	records.append({"validation_section": "national_total",
					"validation_metric": "signed_relative_error",
					"validation_value": float(national_error),
					"n_rows": 1,
					"n_fail_rows": int(abs(national_error) > RELATIVE_ERROR_WARN_THRESHOLD),
					"pass_threshold": RELATIVE_ERROR_PASS_THRESHOLD,
					"warn_threshold": RELATIVE_ERROR_WARN_THRESHOLD,
					"validation_status": _validation_flag(pd.Series([national_error])).iloc[0],
					"interpretation_note": (
						"National total comparison using summed HHR01 regional totals "
						"and summed WEATHERED 2022 HSE-region populations."
					),})

	return pd.DataFrame.from_records(records)


def build_hhr_validation_outputs(inputs: dict[str, pd.DataFrame],
								 validation_year: int = VALIDATION_YEAR,
								 ) -> dict[str, pd.DataFrame]:
	#### Build integrated HHR reference, WEATHERED comparison, and summary validation tables.
	hhr01_reference = clean_hhr01_reference(df=inputs["hhr01_raw"],
										  validation_year=validation_year,)
	hhr02_reference = clean_hhr02_reference(df=inputs["hhr02_raw"],
											validation_year=validation_year,)
	hse_region_sex_shares = clean_hse_region_sex_shares(df=inputs["hse_region_sex_shares"],
														hhr01_reference=hhr01_reference,
														validation_year=validation_year,)
	hhr_implied = build_hhr_implied_age_sex_counts(hhr01_reference=hhr01_reference,
													 hhr02_reference=hhr02_reference,
													 hse_region_sex_shares=hse_region_sex_shares,)
	weathered_regional_totals, weathered_age_sex = build_weathered_hse_population_reference(
		master=inputs["cso_master"],
		validation_year=validation_year,)
	hhr_weathered_validation = build_hhr_weathered_validation(hhr_implied=hhr_implied,
															  weathered_age_sex=weathered_age_sex,)
	hhr_regional_validation = build_hhr_regional_total_validation(
		hhr01_reference=hhr01_reference,
		weathered_regional_totals=weathered_regional_totals,
	)
	summary = build_hhr_validation_summary(hhr_regional_validation=hhr_regional_validation,
										   hhr_weathered_validation=hhr_weathered_validation,)

	return {"hhr_implied_counts": hhr_implied,
			"hhr_weathered_validation": hhr_weathered_validation,
			"hhr_validation_summary": summary,
			"hhr_regional_validation": hhr_regional_validation,}


def get_hhr_validation_output_paths(paths: ProjectPaths) -> dict[str, Path]:
	#### Return output paths for HHR validation EDA tables.
	dirs = paths.stage_dirs("eda")

	return {"hhr_implied_counts": dirs["processed"] / HHR_IMPLIED_OUTPUT_FILE,
			"hhr_weathered_validation": dirs["processed"] / HHR_WEATHERED_VALIDATION_OUTPUT_FILE,
			"hhr_validation_summary": dirs["processed"] / HHR_VALIDATION_SUMMARY_OUTPUT_FILE,}


def remove_legacy_hhr_outputs(paths: ProjectPaths) -> list[Path]:
	#### Remove superseded HHR validation CSV and manifest files.
	dirs = paths.stage_dirs("eda")
	removed = []

	for filename in LEGACY_HHR_OUTPUT_FILES:
		csv_path = dirs["processed"] / filename
		if csv_path.exists():
			csv_path.unlink()
			removed.append(csv_path)

		stem = Path(filename).stem
		for manifest_path in (dirs["manifests"] / f"{stem}.json",
							  dirs["manifests"] / f"{stem}_column_audit.csv"):
			if manifest_path.exists():
				manifest_path.unlink()
				removed.append(manifest_path)

	return removed


def write_hhr_validation_outputs(outputs: dict[str, pd.DataFrame],
								 paths: ProjectPaths,
								 logger,
								 ) -> dict[str, Path]:
	#### Write HHR validation tables to processed EDA outputs and manifests.
	dirs = paths.stage_dirs("eda")
	output_paths = get_hhr_validation_output_paths(paths)
	writable_outputs = {key: outputs[key]
						for key in output_paths
						if key in outputs}

	for key, df in writable_outputs.items():
		output_paths[key] = write_eda_table(df=df,
											output_path=output_paths[key],
											stage=key,
											manifest_dir=dirs["manifests"],
											write_audit=True,
											write_json_manifest=True,)

	log_block(logger,
			  "CSO HHR validation outputs written",
			  {key: str(value) for key, value in output_paths.items()},)

	return output_paths


def run_cso_hhr_validation(paths: ProjectPaths | None = None,
						   validation_year: int = VALIDATION_YEAR,
						   ) -> dict[str, Path]:
	#### Compare WEATHERED 2022 HSE populations with integrated CSO HHR reference counts.
	if paths is None:
		paths = get_project_paths()

	ensure_directories(paths.all_standard_dirs())

	dirs = paths.stage_dirs("eda")

	logger = get_logger(name="cso_hhr_validation",
						log_dir=dirs["logs"],
						filename="cso_hhr_validation.log",)

	logger.info("Starting CSO HHR validation for year %s", validation_year)

	removed_paths = remove_legacy_hhr_outputs(paths=paths)
	if removed_paths:
		log_block(logger,
				  "Removed superseded HHR validation outputs",
				  [str(path) for path in removed_paths],)

	inputs = load_hhr_validation_inputs(paths=paths)
	outputs = build_hhr_validation_outputs(inputs=inputs,
										   validation_year=validation_year,)
	output_paths = write_hhr_validation_outputs(outputs=outputs,
												paths=paths,
												logger=logger,)

	log_block(logger,
			  "CSO HHR validation summary",
			  outputs["hhr_validation_summary"].to_dict(orient="records"),)

	logger.info("Finished CSO HHR validation")

	return output_paths


if __name__ == "__main__":
	run_cso_hhr_validation()
