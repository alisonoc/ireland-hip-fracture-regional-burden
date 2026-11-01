#### preprocess_cso.py

from __future__ import annotations

import unicodedata
from pathlib import Path

import pandas as pd

from crosswalk.paths import ProjectPaths
from crosswalk.preprocessing.common import (collapse_county_to_hse_lookup,
											normalise_county_label)
from crosswalk.utils.io import (
	read_required_csv,
	save_dataset_with_manifest,
)
from crosswalk.utils.logging import get_logger, log_block
from crosswalk.utils.manifest import write_column_audit
from crosswalk.utils.validation import (
	require_columns,
	validate_unique_rows,
)


SEX_KEEP = ["Male", "Female"]

AGE_65PLUS_GROUPS = ["65 - 69 years",
					 "70 - 74 years",
					 "75 - 79 years",
					 "80 - 84 years",
					 "85 years and over",]

PEA08_FILE = "PEA08.20260504T110532.csv"
PEA04_FILE = "PEA04.20260504T110503.csv"

NUTS3_LABEL_MAPPING = {"Midland": "Midlands",}


def normalise_county(series: pd.Series) -> pd.Series:
	#### Normalise county names by trimming, lowercasing, and removing accents.
	ret = series.astype("string").str.strip().str.lower()

	ret = ret.apply(lambda x: unicodedata.normalize("NFKD", x).encode("ascii", "ignore").decode("utf-8")
					if pd.notna(x) else x)

	ret = ret.str.replace("co. ", "", regex=False)
	ret = ret.str.replace("county ", "", regex=False)
	ret = ret.str.replace(r"\s+", " ", regex=True).str.strip()

	ret = normalise_county_label(ret)

	return ret


def load_cso_pea08_population(paths: ProjectPaths) -> pd.DataFrame:
	#### Load CSO PEA08 county age-sex population estimates.
	path = paths.raw / PEA08_FILE

	ret = read_required_csv(path=path,
							name="CSO PEA08 population",)

	return ret


def load_cso_pea04_population(paths: ProjectPaths) -> pd.DataFrame:
	#### Load CSO PEA04 province/NUTS3 age-sex population estimates.
	path = paths.raw / PEA04_FILE

	ret = read_required_csv(path=path,
							name="CSO PEA04 population",)

	return ret


def load_county_to_region(dirs: dict[str, Path]) -> pd.DataFrame:
	#### Load county-to-region baseline mapping from preprocessing outputs.
	path = dirs["processed"] / "geo_county_to_region_baseline.csv"

	df = read_required_csv(path=path,
						   name="County-to-region baseline",)

	required = {"county_english",
				"hse_region",
				"small_area_weight",
				"nuts3_names",}

	require_columns(df=df,
					required=required,
					name="county-to-region baseline",)

	ret = df[["county_english",
			  "hse_region",
			  "small_area_weight",
			  "nuts3_names",]].copy()

	ret["county_english"] = normalise_county(ret["county_english"])
	ret["hse_region"] = ret["hse_region"].astype("string").str.strip()
	ret["nuts3_names"] = ret["nuts3_names"].astype("string").str.strip()
	ret["nuts3_names"] = ret["nuts3_names"].replace(NUTS3_LABEL_MAPPING)
	ret["small_area_weight"] = pd.to_numeric(ret["small_area_weight"], errors="coerce")

	if ret["small_area_weight"].isna().any():
		raise ValueError("County-to-region baseline contains missing or invalid small_area_weight values.")

	weight_check = (ret.groupby("county_english",
								as_index=False,
								dropna=False,
								observed=True,)
					.agg(weight_sum=("small_area_weight", "sum"),))

	bad_weights = weight_check.loc[~weight_check["weight_sum"].round(6).eq(1)].copy()

	if not bad_weights.empty:
		raise ValueError(f"County-to-region weights do not sum to 1: {bad_weights.to_dict(orient='records')}")

	return ret


def clean_cso_pea08_population(df: pd.DataFrame) -> pd.DataFrame:
	#### Clean CSO PEA08 county age-sex population estimates.
	ret = df.copy()

	ret = ret.rename(columns={"Year": "year",
							  "Age Group": "age_group",
							  "Sex": "sex",
							  "County": "county_english",
							  "VALUE": "county_population",})

	required = {"year",
				"age_group",
				"sex",
				"county_english",
				"county_population",}

	require_columns(df=ret,
					required=required,
					name="PEA08 population after rename",)

	ret["year"] = pd.to_numeric(ret["year"], errors="coerce").astype("Int64")
	ret["county_population"] = pd.to_numeric(ret["county_population"], errors="coerce") * 1000
	ret["age_group"] = ret["age_group"].astype("string").str.strip()
	ret["sex"] = ret["sex"].astype("string").str.strip()
	ret["county_english"] = normalise_county(ret["county_english"])

	ret = ret.loc[ret["sex"].isin(SEX_KEEP)].copy()
	ret = ret.loc[ret["county_english"].ne("ireland")].copy()
	ret = ret.dropna(subset=["year",
							 "age_group",
							 "sex",
							 "county_english",
							 "county_population",])

	ret["population_source"] = "pea08_county_age_sex_observed"

	ret = ret[["year",
			   "age_group",
			   "sex",
			   "county_english",
			   "county_population",
			   "population_source",]].copy()

	return ret


def clean_cso_pea04_population(df: pd.DataFrame) -> pd.DataFrame:
	#### Clean CSO PEA04 NUTS3 age-sex population estimates.
	ret = df.copy()

	ret = ret.rename(columns={"Year": "year",
							  "Age Group": "age_group",
							  "Sex": "sex",
							  "Region": "nuts3_names",
							  "VALUE": "area_population",})

	required = {"year",
				"age_group",
				"sex",
				"nuts3_names",
				"area_population",}

	require_columns(df=ret,
					required=required,
					name="PEA04 population after rename",)

	ret["year"] = pd.to_numeric(ret["year"], errors="coerce").astype("Int64")
	ret["area_population"] = pd.to_numeric(ret["area_population"], errors="coerce") * 1000
	ret["age_group"] = ret["age_group"].astype("string").str.strip()
	ret["sex"] = ret["sex"].astype("string").str.strip()
	ret["nuts3_names"] = ret["nuts3_names"].astype("string").str.strip()
	ret["nuts3_names"] = ret["nuts3_names"].replace(NUTS3_LABEL_MAPPING)

	ret = ret.loc[ret["sex"].isin(SEX_KEEP)].copy()
	ret = ret.loc[ret["nuts3_names"].ne("State")].copy()
	ret = ret.dropna(subset=["year",
							 "age_group",
							 "sex",
							 "nuts3_names",
							 "area_population",])

	ret["population_source"] = "pea04_nuts3_age_sex_observed"

	ret = ret[["year",
			   "age_group",
			   "sex",
			   "nuts3_names",
			   "area_population",
			   "population_source",]].copy()

	return ret


def check_nuts3_county_mapping(nuts3_to_county: pd.DataFrame) -> None:
	#### Check that each county maps to exactly one NUTS3 name.
	check = (nuts3_to_county.groupby("county_english",
									as_index=False,
									dropna=False,
									observed=True,)
			 .agg(n_nuts3=("nuts3_names", "nunique"),))

	bad = check.loc[check["n_nuts3"].ne(1)].copy()

	if not bad.empty:
		raise ValueError(f"Counties mapped to multiple NUTS3 names: {bad.to_dict(orient='records')}")


def add_nuts3_to_pea08(pea08: pd.DataFrame,
					   nuts3_to_county: pd.DataFrame,
					   ) -> pd.DataFrame:
	#### Add NUTS3 labels to observed PEA08 county population rows.
	ret = pea08.merge(nuts3_to_county,
					  on="county_english",
					  how="left",
					  validate="many_to_one",)

	if ret["nuts3_names"].isna().any():
		missing = sorted(ret.loc[ret["nuts3_names"].isna(), "county_english"]
						 .dropna()
						 .unique()
						 .tolist())

		raise ValueError(f"PEA08 counties missing NUTS3 mapping: {missing}")

	return ret


def expand_pea04_to_counties(pea04: pd.DataFrame,
							 nuts3_to_county: pd.DataFrame,
							 ) -> pd.DataFrame:
	#### Expand PEA04 NUTS3 population rows to county rows before share allocation.
	ret = pea04.merge(nuts3_to_county,
					  on="nuts3_names",
					  how="left",
					  validate="many_to_many",)

	if ret["county_english"].isna().any():
		missing = sorted(ret.loc[ret["county_english"].isna(), "nuts3_names"]
						 .dropna()
						 .unique()
						 .tolist())

		raise ValueError(f"PEA04 NUTS3 regions missing county mapping: {missing}")

	return ret


def build_nuts3_population_check(pea04_expanded: pd.DataFrame,
								 pea08_mapped: pd.DataFrame,
								 logger,
								 ) -> pd.DataFrame:
	#### Compare overlapping PEA04 and PEA08 NUTS3 population totals.
	pea08_nuts3_totals = (pea08_mapped.groupby(["year",
												"age_group",
												"sex",
												"nuts3_names",],
											   as_index=False,
											   dropna=False,
											   observed=True,)
						  .agg(pea08_nuts3_population=("county_population", "sum"),))

	pea04_nuts3_totals = (pea04_expanded[["year",
										  "age_group",
										  "sex",
										  "nuts3_names",
										  "area_population",]]
						  .drop_duplicates()
						  .rename(columns={"area_population": "pea04_nuts3_population",}))

	ret = pea04_nuts3_totals.merge(pea08_nuts3_totals,
								   on=["year",
									   "age_group",
									   "sex",
									   "nuts3_names",],
								   how="inner",
								   validate="one_to_one",)

	ret["difference"] = ret["pea08_nuts3_population"] - ret["pea04_nuts3_population"]
	ret["relative_difference"] = ret["difference"] / ret["pea04_nuts3_population"]

	large_diff = ret.loc[ret["relative_difference"].abs().ge(0.05)].copy()

	if not large_diff.empty:
		logger.warning("PEA04 and PEA08 NUTS3 population consistency check has %s rows exceeding 5%% relative difference",
					   int(large_diff.shape[0]),)

	log_block(logger,
			  "CSO PEA04 and PEA08 population consistency check",
			  {"pea04_min_year": int(pea04_expanded["year"].min()),
			   "pea04_max_year": int(pea04_expanded["year"].max()),
			   "pea08_min_year": int(pea08_mapped["year"].min()),
			   "pea08_max_year": int(pea08_mapped["year"].max()),
			   "comparison_rows": int(ret.shape[0]),
			   "max_abs_difference": float(ret["difference"].abs().max()),
			   "max_abs_relative_difference": float(ret["relative_difference"].abs().max()),},)

	return ret


def build_county_shares(pea08_mapped: pd.DataFrame) -> pd.DataFrame:
	#### Build mean county population shares within each NUTS3 area using PEA08 all-age rows.
	pea08_share_base = pea08_mapped.loc[pea08_mapped["age_group"].eq("All ages")].copy()

	if pea08_share_base.empty:
		raise ValueError("PEA08 contains no 'All ages' rows for county-share estimation.")

	pea08_total = (pea08_share_base.groupby(["year",
											 "nuts3_names",
											 "county_english",],
											as_index=False,
											dropna=False,
											observed=True,)
				   .agg(county_population=("county_population", "sum"),))

	pea08_total["nuts3_total"] = pea08_total.groupby(["year",
													  "nuts3_names",],
													 observed=True,)["county_population"].transform("sum")

	pea08_total["county_share"] = pea08_total["county_population"] / pea08_total["nuts3_total"]

	ret = (pea08_total.groupby(["nuts3_names",
							   "county_english",],
							  as_index=False,
							  dropna=False,
							  observed=True,)
		   .agg(county_share=("county_share", "mean"),))

	share_check = (ret.groupby("nuts3_names",
							   observed=True,)["county_share"]
				   .sum()
				   .round(6))

	if not share_check.eq(1.0).all():
		bad_shares = share_check.loc[share_check.ne(1.0)]
		raise ValueError(f"County shares do not sum to 1 within NUTS3: {bad_shares.to_dict()}")

	return ret


def build_pea04_county_estimates(pea04_expanded: pd.DataFrame,
								 county_shares: pd.DataFrame,
								 ) -> pd.DataFrame:
	#### Estimate county population from PEA04 NUTS3 population using PEA08 county shares.
	ret = pea04_expanded.merge(county_shares,
							   on=["nuts3_names",
								   "county_english",],
							   how="left",
							   validate="many_to_one",)

	if ret["county_share"].isna().any():
		missing = (ret.loc[ret["county_share"].isna(), ["nuts3_names", "county_english"]]
				   .drop_duplicates()
				   .head(20))

		raise ValueError(f"Missing county shares for PEA04 expansion: {missing.to_dict(orient='records')}")

	ret["county_population"] = ret["area_population"] * ret["county_share"]
	ret["population_source"] = "pea04_nuts3_population_x_pea08_county_share"
	ret["is_interpolated"] = True

	ret = ret[["year",
			   "age_group",
			   "sex",
			   "nuts3_names",
			   "county_english",
			   "county_population",
			   "population_source",
			   "is_interpolated",]].copy()

	return ret


def combine_county_population(pea04_county_estimated: pd.DataFrame,
							  pea08_mapped: pd.DataFrame,
							  ) -> pd.DataFrame:
	#### Combine estimated PEA04 county rows with observed PEA08 county rows.
	pea08_observed = pea08_mapped[["year",
								   "age_group",
								   "sex",
								   "nuts3_names",
								   "county_english",
								   "county_population",
								   "population_source",]].copy()

	pea08_observed["is_interpolated"] = False

	ret = pd.concat([pea04_county_estimated,
					 pea08_observed,],
					ignore_index=True,)

	ret["source_priority"] = ret["population_source"].map({"pea08_county_age_sex_observed": 1,
														   "pea04_nuts3_population_x_pea08_county_share": 2,})

	ret = (ret.sort_values(["year",
							"age_group",
							"sex",
							"county_english",
							"source_priority",])
		   .drop_duplicates(subset=["year",
									"age_group",
									"sex",
									"county_english",],
							keep="first")
		   .drop(columns="source_priority")
		   .reset_index(drop=True))

	return ret


def allocate_county_population_to_hse_region(county_population: pd.DataFrame,
											 county_to_hse: pd.DataFrame,
											 ) -> pd.DataFrame:
	#### Allocate county population to HSE regions using small-area weights.
	ret = county_population.merge(county_to_hse,
								  on="county_english",
								  how="left",
								  validate="many_to_many",)

	if ret["hse_region"].isna().any():
		missing = sorted(ret.loc[ret["hse_region"].isna(), "county_english"]
						 .dropna()
						 .unique()
						 .tolist())

		raise ValueError(f"County populations missing HSE region mapping: {missing}")

	ret["population"] = ret["county_population"] * ret["small_area_weight"]

	ret["population_method"] = ret["population_source"] + "_x_small_area_hse_region_weight"

	ret = ret[["year",
			   "age_group",
			   "sex",
			   "county_english",
			   "hse_region",
			   "nuts3_names",
			   "county_population",
			   "small_area_weight",
			   "population",
			   "is_interpolated",
			   "population_source",
			   "population_method",]].copy()

	return ret


def build_expanded_cso_population_dataset(pea04: pd.DataFrame,
										  pea08: pd.DataFrame,
										  county_to_region: pd.DataFrame,
										  paths: ProjectPaths,
										  logger,
										  ) -> dict[str, pd.DataFrame]:
	#### Build county, HSE-region, NUTS3 age-sex population using PEA08 observed rows and PEA04 backfill.
	logger.info("Building expanded CSO population dataset from PEA04 and PEA08")

	county_to_region = county_to_region.copy()
	county_to_region["county_english"] = normalise_county(county_to_region["county_english"])

	county_to_hse = collapse_county_to_hse_lookup(county_to_region)

	nuts3_to_county = (county_to_region[["county_english", "nuts3_names"]]
					   .drop_duplicates()
					   .copy())

	check_nuts3_county_mapping(nuts3_to_county)

	pea08_mapped = add_nuts3_to_pea08(pea08=pea08,
									  nuts3_to_county=nuts3_to_county,)

	pea04_expanded = expand_pea04_to_counties(pea04=pea04,
											   nuts3_to_county=nuts3_to_county,)

	nuts3_population_check = build_nuts3_population_check(pea04_expanded=pea04_expanded,
														  pea08_mapped=pea08_mapped,
														  logger=logger,)

	county_shares = build_county_shares(pea08_mapped)

	pea04_county_estimated = build_pea04_county_estimates(pea04_expanded=pea04_expanded,
														  county_shares=county_shares,)

	county_population = combine_county_population(pea04_county_estimated=pea04_county_estimated,
												  pea08_mapped=pea08_mapped,)

	hse_population = allocate_county_population_to_hse_region(county_population=county_population,
															  county_to_hse=county_to_hse,)

	ret = {"master_population": hse_population,
		   "pea04_county_estimated": pea04_county_estimated,
		   "county_population_combined": county_population,
		   "county_shares": county_shares,
		   "nuts3_population_check": nuts3_population_check,}

	return ret


def filter_master_population(master: pd.DataFrame,
							 age_groups: str | list[str] | None = None,
							 ) -> pd.DataFrame:
	#### Filter master population dataset for modelling-specific age-group extracts.
	ret = master.copy()

	if age_groups is None:
		return ret

	if isinstance(age_groups, str):
		age_groups = [age_groups]

	missing = sorted(set(age_groups) - set(ret["age_group"].dropna().unique()))

	if missing:
		raise ValueError(f"Requested age groups not found in master population dataset: {missing}")

	ret = ret.loc[ret["age_group"].isin(age_groups)].copy()

	ret = ret[["year",
			   "age_group",
			   "sex",
			   "county_english",
			   "hse_region",
			   "nuts3_names",
			   "population",
			   "is_interpolated",
			   "population_source",]].copy()

	return ret


def run_cso_preprocessing(paths: ProjectPaths) -> dict[str, Path]:
	#### Run CSO preprocessing and save county-region age-sex population outputs.
	stage = paths.stage_from_module(__name__)
	dirs = paths.stage_dirs(stage)

	logger = get_logger(name="preprocess_cso",
						log_dir=dirs["logs"],
						filename="preprocess_cso.log",)

	logger.info("Starting CSO preprocessing")

	log_block(logger,
			  "CSO population source files",
			  {"PEA08_source": "CSO population estimates / annual population estimates / PEA08 - estimated population (Persons in April)",
			   "PEA08_coverage": "2017-2025 by age, sex and county",
			   "PEA08_file": PEA08_FILE,
			   "PEA04_source": "CSO population estimates / annual population estimates / PEA04 - estimated population (Persons in April)",
			   "PEA04_coverage": "2011-2025 by age, sex and province/NUTS3 region",
			   "PEA04_file": PEA04_FILE,},)

	pea08_raw = load_cso_pea08_population(paths)
	pea04_raw = load_cso_pea04_population(paths)
	county_to_region = load_county_to_region(dirs)

	write_column_audit(pea08_raw,
					   dirs["manifests"] / "cso_pea08_raw_column_audit.csv",)

	write_column_audit(pea04_raw,
					   dirs["manifests"] / "cso_pea04_raw_column_audit.csv",)

	pea08 = clean_cso_pea08_population(pea08_raw)
	pea04 = clean_cso_pea04_population(pea04_raw)

	population_outputs = build_expanded_cso_population_dataset(pea04=pea04,
															   pea08=pea08,
															   county_to_region=county_to_region,
															   paths=paths,
															   logger=logger,)

	master_population = population_outputs["master_population"]

	region_65plus = filter_master_population(master=master_population,
											  age_groups=AGE_65PLUS_GROUPS,)

	validate_unique_rows(df=master_population,
						 keys=["year",
							   "age_group",
							   "sex",
							   "hse_region",
							   "county_english",],
						 name="CSO master population dataset",)

	validate_unique_rows(df=region_65plus,
						 keys=["year",
							   "age_group",
							   "sex",
							   "hse_region",
							   "county_english",],
						 name="CSO 65+ region-county population extract",)

	outputs = {"pea08_county_age_sex": dirs["processed"] / "cso_pea08_county_age_sex_population.csv",
			   "pea04_nuts3_age_sex": dirs["processed"] / "cso_pea04_nuts3_age_sex_population.csv",
			   "pea04_county_estimated": dirs["processed"] / "cso_pea04_county_population_estimated.csv",
			   "county_population_combined": dirs["processed"] / "cso_county_population_combined_pea04_pea08.csv",
			   "county_shares": dirs["processed"] / "cso_pea08_county_share_within_nuts3.csv",
			   "nuts3_population_check": dirs["processed"] / "cso_pea04_pea08_nuts3_population_check.csv",
			   "master_population": dirs["processed"] / "cso_population_master_age_sex_region_county_year.csv",
			   "region_65plus": dirs["processed"] / "cso_population_65plus_region_county_year_sex.csv",}

	save_dataset_with_manifest(df=pea08,
							   output_path=outputs["pea08_county_age_sex"],
							   manifest_dir=dirs["manifests"],
							   stage="cso_pea08_county_age_sex_population",)

	save_dataset_with_manifest(df=pea04,
							   output_path=outputs["pea04_nuts3_age_sex"],
							   manifest_dir=dirs["manifests"],
							   stage="cso_pea04_nuts3_age_sex_population",)

	save_dataset_with_manifest(df=population_outputs["pea04_county_estimated"],
							   output_path=outputs["pea04_county_estimated"],
							   manifest_dir=dirs["manifests"],
							   stage="cso_pea04_county_population_estimated",)

	save_dataset_with_manifest(df=population_outputs["county_population_combined"],
							   output_path=outputs["county_population_combined"],
							   manifest_dir=dirs["manifests"],
							   stage="cso_county_population_combined_pea04_pea08",)

	save_dataset_with_manifest(df=population_outputs["county_shares"],
							   output_path=outputs["county_shares"],
							   manifest_dir=dirs["manifests"],
							   stage="cso_pea08_county_share_within_nuts3",)

	save_dataset_with_manifest(df=population_outputs["nuts3_population_check"],
							   output_path=outputs["nuts3_population_check"],
							   manifest_dir=dirs["manifests"],
							   stage="cso_pea04_pea08_nuts3_population_check",)

	save_dataset_with_manifest(df=master_population,
							   output_path=outputs["master_population"],
							   manifest_dir=dirs["manifests"],
							   stage="cso_population_master_age_sex_region_county_year",)

	save_dataset_with_manifest(df=region_65plus,
							   output_path=outputs["region_65plus"],
							   manifest_dir=dirs["manifests"],
							   stage="cso_population_65plus_region_county_year_sex",)

	log_block(logger,
			  "CSO preprocessing outputs saved",
			  {key: str(value) for key, value in outputs.items()},)

	logger.info("Finished CSO preprocessing")

	return outputs