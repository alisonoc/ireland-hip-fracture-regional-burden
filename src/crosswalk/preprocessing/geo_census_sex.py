#### geo_census_sex.py

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from crosswalk.paths import ProjectPaths
from crosswalk.preprocessing.geo_common import (
	load_county_gaeilge_to_english_mapping,
	normalise_county_name,
	normalise_region_name,
)
from crosswalk.preprocessing.preprocess_cso_projections import find_latest_raw_file
from crosswalk.preprocessing.reference import order_hse_region_rows
from crosswalk.utils.io import save_dataset_with_manifest
from crosswalk.utils.logging import log_block
from crosswalk.utils.manifest import write_column_audit
from crosswalk.utils.validation import (
	require_columns,
	validate_non_empty,
	validate_no_missing,
	validate_non_negative,
)


CENSUS_TABLE_12_SA_PATTERN = "CensusHub2022_T1_2_SA*.geojson"
CENSUS_2022_YEAR = 2022
CENSUS_2022_NATIONAL_TOTAL = 5_149_139
CENSUS_2022_NATIONAL_MALE = 2_544_549
CENSUS_2022_NATIONAL_FEMALE = 2_604_590
CENSUS_TOTAL_TOLERANCE = 0.001

GEO_CENSUS_2022_SMALL_AREA_SEX_FILE = "geo_census_2022_small_area_sex.csv"
GEO_CENSUS_2022_COUNTY_AREA_SEX_FILE = "geo_census_2022_county_area_sex.csv"
GEO_CENSUS_2022_HSE_REGION_SEX_SHARES_FILE = "geo_census_2022_hse_region_sex_shares.csv"

CENSUS_SEX_SOURCE_DATASET = "CensusHub2022 Table 1.2 small-area population by sex and marital status"


def load_census_table_12_small_area_sex(paths: ProjectPaths) -> gpd.GeoDataFrame:
	#### Load Census 2022 Table 1.2 small-area sex counts from GeoHive GeoJSON.
	census_path = find_latest_raw_file(paths=paths,
									   pattern=CENSUS_TABLE_12_SA_PATTERN,)

	ret = gpd.read_file(census_path)

	require_columns(df=ret,
					required={"GUID",
							  "SA_PUB2022",
							  "T1_2TM",
							  "T1_2TF",
							  "T1_2T"},
					name=f"Census Table 1.2 small-area sex file {census_path.name}",)

	return ret


def clean_census_table_12_small_area_sex(df: gpd.GeoDataFrame,
										 year: int = CENSUS_2022_YEAR,
										 ) -> pd.DataFrame:
	#### Clean Census 2022 small-area male and female population counts.
	ret = df.copy()

	ret = ret.rename(columns={"GUID": "SA_GUID_2022",
							  "T1_2TM": "census_male",
							  "T1_2TF": "census_female",
							  "T1_2T": "census_total",})

	ret["year"] = year
	ret["SA_GUID_2022"] = ret["SA_GUID_2022"].astype("string").str.strip()
	ret["SA_PUB2022"] = ret["SA_PUB2022"].astype("string").str.strip()

	for column in ["census_male", "census_female", "census_total"]:
		ret[column] = pd.to_numeric(ret[column], errors="coerce")

	ret = ret[["year", "SA_GUID_2022", "SA_PUB2022", "census_male", "census_female", "census_total"]].drop_duplicates()

	validate_non_empty(df=ret,
					   name="Census Table 1.2 cleaned small-area sex rows",)

	validate_no_missing(df=ret,
						columns=["year",
								 "SA_GUID_2022",
								 "SA_PUB2022",
								 "census_male",
								 "census_female",
								 "census_total"],
						name="Census Table 1.2 cleaned small-area sex",)

	validate_non_negative(df=ret,
						  columns=["census_male", "census_female", "census_total"],
						  name="Census Table 1.2 cleaned small-area sex",)

	if not ret["SA_GUID_2022"].is_unique:
		raise ValueError("Census Table 1.2 small-area sex table contains duplicate SA_GUID_2022 values.")

	sex_total_check = (ret["census_male"] + ret["census_female"] - ret["census_total"]).abs()

	if sex_total_check.gt(0).any():
		examples = ret.loc[sex_total_check.gt(0), ["SA_GUID_2022", "census_male", "census_female", "census_total"]].head(10)
		raise ValueError(f"Census male plus female does not equal total for some small areas: {examples.to_dict(orient='records')}")

	national_total = float(ret["census_total"].sum())
	relative_gap = abs(national_total - CENSUS_2022_NATIONAL_TOTAL) / CENSUS_2022_NATIONAL_TOTAL

	if relative_gap > CENSUS_TOTAL_TOLERANCE:
		raise ValueError(
			f"Census Table 1.2 national total {national_total:,.0f} differs from expected "
			f"{CENSUS_2022_NATIONAL_TOTAL:,} by {relative_gap:.4f}"
		)

	ret["male_share"] = ret["census_male"] / ret["census_total"].replace(0, pd.NA)
	ret["female_share"] = ret["census_female"] / ret["census_total"].replace(0, pd.NA)
	ret["population_source"] = CENSUS_SEX_SOURCE_DATASET

	return ret.sort_values(["SA_PUB2022"]).reset_index(drop=True)


def attach_small_area_geography(census_small_area_sex: pd.DataFrame,
								small_area_regions: pd.DataFrame,
								paths: ProjectPaths,
								) -> pd.DataFrame:
	#### Attach canonical county and HSE-region geography to Census small-area sex counts.
	require_columns(df=small_area_regions,
					required={"SA_GUID_2022",
							  "COUNTY_ENGLISH",
							  "COUNTY_GAEILGE",
							  "HR_operational_name"},
					name="small_area_regions for Census sex geography join",)

	geography = small_area_regions[["SA_GUID_2022",
									"COUNTY_ENGLISH",
									"COUNTY_GAEILGE",
									"HR_operational_name"]].copy()

	geography["SA_GUID_2022"] = geography["SA_GUID_2022"].astype("string").str.strip()
	geography["county_area"] = normalise_county_name(geography["COUNTY_ENGLISH"])
	geography["county_gaeilge"] = geography["COUNTY_GAEILGE"].astype("string").str.strip()
	geography["hse_region"] = normalise_region_name(geography["HR_operational_name"])

	county_english_mapping = load_county_gaeilge_to_english_mapping(paths)
	geography["county_english"] = geography["county_gaeilge"].map(county_english_mapping)

	if geography["county_english"].isna().any():
		missing = sorted(geography.loc[geography["county_english"].isna(), "county_gaeilge"].dropna().unique().tolist())
		raise ValueError(f"Census sex geography join has unmapped county_gaeilge values: {missing}")

	geography = geography.drop(columns=["COUNTY_ENGLISH", "COUNTY_GAEILGE", "HR_operational_name"]).drop_duplicates()

	ret = census_small_area_sex.merge(geography,
									  on="SA_GUID_2022",
									  how="inner",
									  validate="one_to_one",)

	if ret.shape[0] != census_small_area_sex.shape[0]:
		missing = census_small_area_sex.loc[~census_small_area_sex["SA_GUID_2022"].isin(ret["SA_GUID_2022"]), "SA_GUID_2022"]
		raise ValueError(
			f"Census small-area sex rows missing geography after join: {missing.head(10).tolist()}"
		)

	ret["interpretation_note"] = (
		"Census 2022 small-area male and female counts from Table 1.2, joined to WEATHERED "
		"small-area county and HSE-region geography."
	)

	return ret.sort_values(["hse_region", "county_area", "SA_PUB2022"]).reset_index(drop=True)


def build_census_2022_county_area_sex(census_small_area_sex: pd.DataFrame,
									  year: int = CENSUS_2022_YEAR,
									  ) -> pd.DataFrame:
	#### Aggregate Census 2022 small-area sex counts to county-area level.
	required = {"year",
				"county_gaeilge",
				"county_english",
				"county_area",
				"census_male",
				"census_female",
				"census_total"}

	require_columns(df=census_small_area_sex,
					required=required,
					name="Census small-area sex for county-area aggregation",)

	ret = (census_small_area_sex.groupby(["year",
										 "county_gaeilge",
										 "county_english",
										 "county_area"],
										as_index=False,
										observed=True,)
		   .agg(census_male=("census_male", "sum"),
				census_female=("census_female", "sum"),
				census_total=("census_total", "sum"),
				n_small_areas=("SA_GUID_2022", "nunique"),))

	validate_non_empty(df=ret,
					   name="Census 2022 county-area sex rows",)

	ret["male_share"] = ret["census_male"] / (ret["census_male"] + ret["census_female"])
	ret["female_share"] = ret["census_female"] / (ret["census_male"] + ret["census_female"])
	ret["population_source"] = CENSUS_SEX_SOURCE_DATASET
	ret["interpretation_note"] = (
		"Census 2022 county-area male and female counts aggregated from Table 1.2 small-area sex counts."
	)

	return ret.sort_values(["county_gaeilge", "county_area"]).reset_index(drop=True)


def build_census_2022_hse_region_sex_shares(census_small_area_sex: pd.DataFrame,
											year: int = CENSUS_2022_YEAR,
											) -> pd.DataFrame:
	#### Aggregate Census 2022 small-area sex counts to HSE-region level.
	require_columns(df=census_small_area_sex,
					required={"year",
							  "hse_region",
							  "census_male",
							  "census_female",
							  "census_total"},
					name="Census small-area sex for HSE-region aggregation",)

	ret = (census_small_area_sex.groupby(["year", "hse_region"],
										as_index=False,
										observed=True,)
		   .agg(census_male=("census_male", "sum"),
				census_female=("census_female", "sum"),
				census_total=("census_total", "sum"),
				n_small_areas=("SA_GUID_2022", "nunique"),))

	validate_non_empty(df=ret,
					   name="Census 2022 HSE-region sex rows",)

	ret["male_share"] = ret["census_male"] / (ret["census_male"] + ret["census_female"])
	ret["female_share"] = ret["census_female"] / (ret["census_male"] + ret["census_female"])
	ret["population_source"] = CENSUS_SEX_SOURCE_DATASET
	ret["interpretation_note"] = (
		"Census 2022 HSE-region male and female counts and sex shares aggregated from "
		"Table 1.2 small-area sex counts using WEATHERED small-area to HSE-region geography."
	)

	return order_hse_region_rows(ret)


def build_census_2022_sex_outputs(census_raw: gpd.GeoDataFrame,
								  small_area_regions: pd.DataFrame,
								  paths: ProjectPaths,
								  year: int = CENSUS_2022_YEAR,
								  ) -> dict[str, pd.DataFrame]:
	#### Build Census 2022 sex-count tables at small-area, county-area, and HSE-region levels.
	census_small_area_sex = clean_census_table_12_small_area_sex(df=census_raw,
																  year=year,)
	census_small_area_sex = attach_small_area_geography(census_small_area_sex=census_small_area_sex,
														small_area_regions=small_area_regions,
														paths=paths,)
	census_county_area_sex = build_census_2022_county_area_sex(census_small_area_sex=census_small_area_sex,
															   year=year,)
	census_hse_region_sex_shares = build_census_2022_hse_region_sex_shares(
		census_small_area_sex=census_small_area_sex,
		year=year,
	)

	return {"census_small_area_sex": census_small_area_sex,
			"census_county_area_sex": census_county_area_sex,
			"census_hse_region_sex_shares": census_hse_region_sex_shares,}


def write_census_2022_sex_outputs(outputs: dict[str, pd.DataFrame],
								  dirs: dict[str, Path],
								  logger,
								  ) -> dict[str, Path]:
	#### Write Census 2022 sex-count tables and manifests to processed geography outputs.
	output_specs = {"census_small_area_sex": (GEO_CENSUS_2022_SMALL_AREA_SEX_FILE,
											  "geo_census_2022_small_area_sex"),
					"census_county_area_sex": (GEO_CENSUS_2022_COUNTY_AREA_SEX_FILE,
											   "geo_census_2022_county_area_sex"),
					"census_hse_region_sex_shares": (GEO_CENSUS_2022_HSE_REGION_SEX_SHARES_FILE,
													  "geo_census_2022_hse_region_sex_shares"),}

	output_paths = {}

	for key, (filename, stage) in output_specs.items():
		output_paths[key] = save_dataset_with_manifest(df=outputs[key],
													   output_path=dirs["processed"] / filename,
													   manifest_dir=dirs["manifests"],
													   stage=stage,)

	log_block(logger,
			  "Census 2022 sex-count outputs saved",
			  {key: str(path) for key, path in output_paths.items()},)

	return output_paths


def write_census_2022_sex_raw_audit(dirs: dict[str, Path],
									logger,
									census_raw: gpd.GeoDataFrame,
									) -> None:
	#### Write raw Census Table 1.2 small-area sex input audit.
	audit_path = dirs["manifests"] / "geo_census_table_12_small_area_raw_column_audit.csv"

	write_column_audit(census_raw.drop(columns="geometry", errors="ignore"),
					   audit_path,)

	log_block(logger,
			  "Census Table 1.2 raw input audit saved",
			  {"audit_path": str(audit_path)},)


def run_census_2022_sex_preprocessing(small_area_regions: pd.DataFrame,
									  paths: ProjectPaths,
									  dirs: dict[str, Path],
									  logger,
									  year: int = CENSUS_2022_YEAR,
									  ) -> dict[str, Path]:
	#### Load Census 2022 Table 1.2 and write small-area, county-area, and HSE-region sex outputs.
	census_raw = load_census_table_12_small_area_sex(paths=paths)

	write_census_2022_sex_raw_audit(dirs=dirs,
									logger=logger,
									census_raw=census_raw,)

	outputs = build_census_2022_sex_outputs(census_raw=census_raw,
											small_area_regions=small_area_regions,
											paths=paths,
											year=year,)

	log_block(logger,
			  "Census 2022 sex-count summary",
			  {"n_small_areas": int(outputs["census_small_area_sex"].shape[0]),
			   "n_county_areas": int(outputs["census_county_area_sex"].shape[0]),
			   "n_hse_regions": int(outputs["census_hse_region_sex_shares"].shape[0]),
			   "national_male": float(outputs["census_hse_region_sex_shares"]["census_male"].sum()),
			   "national_female": float(outputs["census_hse_region_sex_shares"]["census_female"].sum()),
			   "national_total": float(outputs["census_hse_region_sex_shares"]["census_total"].sum()),},)

	return write_census_2022_sex_outputs(outputs=outputs,
									   dirs=dirs,
									   logger=logger,)
