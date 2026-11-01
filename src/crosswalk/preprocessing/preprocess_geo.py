#### preprocess_geo.py

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from crosswalk.paths import ProjectPaths
from crosswalk.utils.io import (
	ensure_directories,
	save_dataset_with_manifest,
)
from crosswalk.utils.logging import get_logger, log_block
from crosswalk.utils.manifest import write_column_audit
from crosswalk.utils.schema import apply_schema
from crosswalk.utils.validation import (
	require_columns,
)
from crosswalk.preprocessing.geo_common import (
	load_county_gaeilge_to_english_mapping,
	load_geo_input_schema,
	normalise_county_name,
	normalise_region_name,
)
from crosswalk.preprocessing.geo_census_sex import run_census_2022_sex_preprocessing
from crosswalk.preprocessing.geo_hospitals import write_hospital_geo_outputs

SMALL_AREA_GEOMETRY_FILE = "Small_Area_National_Statistical_Boundaries_2022_Ungeneralised_view_-7354763930310470674.gpkg"
HEALTH_LOOKUP_FILE = "Health_Geographies_(2025)_-_CSO_lookup.geojson"

def build_small_area_county_towns_lookup(small_areas: gpd.GeoDataFrame) -> pd.DataFrame:
	#### Build long lookup of NUTS3 areas, counties, and LEAs from small area preprocessing.
	required = {"SA_NUTS3_NAME",
				"COUNTY_ENGLISH",
				"COUNTY_GAEILGE",
				"CSO_LEA",}

	require_columns(df=small_areas,
					required=required,
					name="small_areas",)

	ret = small_areas[["SA_NUTS3_NAME",
					   "COUNTY_ENGLISH",
					   "COUNTY_GAEILGE",
					   "CSO_LEA",]].dropna(subset=["SA_NUTS3_NAME"]).copy()

	ret["SA_NUTS3_NAME"] = ret["SA_NUTS3_NAME"].astype("string").str.strip()
	ret["COUNTY_ENGLISH"] = ret["COUNTY_ENGLISH"].astype("string").str.strip()
	ret["COUNTY_GAEILGE"] = ret["COUNTY_GAEILGE"].astype("string").str.strip()
	ret["CSO_LEA"] = ret["CSO_LEA"].astype("string").str.strip()

	ret = (ret.drop_duplicates()
		   .sort_values(["SA_NUTS3_NAME", "COUNTY_ENGLISH", "CSO_LEA"])
		   .reset_index(drop=True))

	return ret


def build_county_metadata_lookup(small_areas: gpd.GeoDataFrame) -> pd.DataFrame:
	#### Build county-level metadata from small area preprocessing.
	required = {"COUNTY_ENGLISH",
				"COUNTY_GAEILGE",
				"SA_NUTS3_NAME",
				"CSO_LEA",}

	require_columns(df=small_areas,
					required=required,
					name="small_areas",)

	df = small_areas[["COUNTY_ENGLISH",
					  "COUNTY_GAEILGE",
					  "SA_NUTS3_NAME",
					  "CSO_LEA",]].copy()

	df["county"] = normalise_county_name(df["COUNTY_ENGLISH"])
	df["COUNTY_GAEILGE"] = df["COUNTY_GAEILGE"].astype("string").str.strip()
	df["SA_NUTS3_NAME"] = df["SA_NUTS3_NAME"].astype("string").str.strip()
	df["CSO_LEA"] = df["CSO_LEA"].astype("string").str.strip()

	ret = (df.groupby("county",
					  as_index=False,
					  dropna=False,
					  observed=True,)
		   .agg(county_gaeilge=("COUNTY_GAEILGE", lambda x: "; ".join(sorted(x.dropna().unique()))),
				nuts3_names=("SA_NUTS3_NAME", lambda x: "; ".join(sorted(x.dropna().unique()))),
				lea_names=("CSO_LEA", lambda x: "; ".join(sorted(x.dropna().unique()))),
				n_leas=("CSO_LEA", "nunique"),))

	return ret


def build_county_to_region_baseline(small_area_regions: pd.DataFrame,
									small_areas: gpd.GeoDataFrame,
									paths: ProjectPaths,
									) -> pd.DataFrame:
	#### Build county-to-health-region weights using Gaeilge county identity as canonical county grouping.
	required = {"COUNTY_ENGLISH",
				"COUNTY_GAEILGE",
				"HR_operational_name",
				"SA_GUID_2022",}

	require_columns(df=small_area_regions,
					required=required,
					name="small_area_regions",)

	county_english_mapping = load_county_gaeilge_to_english_mapping(paths)

	crosswalk = (small_area_regions[["COUNTY_ENGLISH",
									 "COUNTY_GAEILGE",
									 "HR_operational_name",
									 "SA_GUID_2022",]]
				 .dropna(subset=["COUNTY_ENGLISH", "COUNTY_GAEILGE", "HR_operational_name"])
				 .copy())

	crosswalk["county_area"] = normalise_county_name(crosswalk["COUNTY_ENGLISH"])
	crosswalk["county_gaeilge"] = crosswalk["COUNTY_GAEILGE"].astype("string").str.strip()
	crosswalk["county_english"] = crosswalk["county_gaeilge"].map(county_english_mapping)
	crosswalk["hse_region"] = normalise_region_name(crosswalk["HR_operational_name"])

	if crosswalk["county_english"].isna().any():
		missing = sorted(crosswalk.loc[crosswalk["county_english"].isna(), "county_gaeilge"]
						 .dropna()
						 .unique()
						 .tolist())

		raise ValueError(f"Missing county Gaeilge to English mappings: {missing}")

	summary = (crosswalk.groupby(["county_gaeilge",
								  "county_english",
								  "county_area",
								  "hse_region",],
								 as_index=False,
								 dropna=False,
								 observed=True,)
			   .agg(n_small_areas=("SA_GUID_2022", "nunique"),))

	canonical_totals = (summary.groupby(["county_gaeilge", "county_english"],
										as_index=False,
										dropna=False,
										observed=True,)
						.agg(total_small_areas=("n_small_areas", "sum"),
							 n_hse_regions=("hse_region", "nunique"),
							 n_county_areas=("county_area", "nunique"),))

	summary = summary.merge(canonical_totals,
							on=["county_gaeilge", "county_english"],
							how="left",
							validate="many_to_one",)

	summary["small_area_weight"] = summary["n_small_areas"] / summary["total_small_areas"]

	county_metadata = build_county_metadata_lookup(small_areas)

	summary = summary.merge(county_metadata.drop(columns=["county_gaeilge"], errors="ignore"),
							left_on="county_area",
							right_on="county",
							how="left",
							validate="many_to_one",)

	summary = summary.drop(columns=["county"], errors="ignore")

	ret = (summary.sort_values(["county_gaeilge",
								"county_english",
								"county_area",
								"hse_region",])
		   .reset_index(drop=True))

	ret = ret[["county_gaeilge",
			   "county_english",
			   "county_area",
			   "hse_region",
			   "nuts3_names",
			   "n_small_areas",
			   "total_small_areas",
			   "n_hse_regions",
			   "n_county_areas",
			   "small_area_weight",]].copy()

	return ret


def validate_county_to_region_baseline(crosswalk: pd.DataFrame) -> None:
	#### Validate county-region weights using Gaeilge county identity.
	required = {"county_gaeilge",
				"county_english",
				"county_area",
				"hse_region",
				"nuts3_names",
				"n_small_areas",
				"total_small_areas",
				"n_hse_regions",
				"n_county_areas",
				"small_area_weight",}

	require_columns(df=crosswalk,
					required=required,
					name="county-region crosswalk",)

	if crosswalk.empty:
		raise ValueError("County-region crosswalk is empty.")

	if crosswalk["county_area"].isna().any():
		raise ValueError("County-region crosswalk contains missing county_area values.")

	if crosswalk["county_gaeilge"].isna().any():
		raise ValueError("County-region crosswalk contains missing county_gaeilge values.")

	if crosswalk["county_english"].isna().any():
		raise ValueError("County-region crosswalk contains missing county_english values.")

	if crosswalk["hse_region"].isna().any():
		raise ValueError("County-region crosswalk contains missing hse_region values.")

	weight_check = (crosswalk.groupby(["county_gaeilge", "county_english"],
									  observed=True,)["small_area_weight"]
					.sum()
					.round(6))

	if not weight_check.eq(1.0).all():
		bad = weight_check.loc[weight_check.ne(1.0)]
		raise ValueError(f"County-region weights do not sum to 1 by county_gaeilge: {bad.to_dict()}")


def load_geo_inputs(paths: ProjectPaths) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
	#### Load and schema-check raw geography inputs.
	small_area_path = paths.raw / SMALL_AREA_GEOMETRY_FILE
	lookup_path = paths.raw / HEALTH_LOOKUP_FILE

	for input_path in [small_area_path, lookup_path]:
		if not input_path.exists():
			raise FileNotFoundError(f"Required geography file not found: {input_path}")

	small_areas = gpd.read_file(small_area_path)
	small_areas = apply_schema(small_areas,
							   load_geo_input_schema(paths=paths,
													 schema_name="small_area"),)

	health_lookup = gpd.read_file(lookup_path)
	health_lookup = apply_schema(health_lookup,
								 load_geo_input_schema(paths=paths,
													   schema_name="health_lookup"),)

	return small_areas, health_lookup


def write_geo_audits(dirs: dict[str, Path],
					 logger,
					 small_areas: gpd.GeoDataFrame,
					 health_lookup: gpd.GeoDataFrame,
					 ) -> None:
	#### Write raw geography column audits.
	small_areas_audit_path = dirs["manifests"] / "geo_small_areas_raw_column_audit.csv"
	health_lookup_audit_path = dirs["manifests"] / "geo_health_lookup_raw_column_audit.csv"

	write_column_audit(small_areas.drop(columns="geometry", errors="ignore"),
					   small_areas_audit_path,)

	write_column_audit(health_lookup.drop(columns="geometry", errors="ignore"),
					   health_lookup_audit_path,)

	log_block(logger,
			  "Raw GEO column audits saved",
			  {"small_areas": str(small_areas_audit_path),
			   "health_lookup": str(health_lookup_audit_path),},)


def build_health_regions(small_areas: gpd.GeoDataFrame,
						 health_lookup: gpd.GeoDataFrame,
						 ) -> pd.DataFrame:
	#### Build small-area table with HSE health-region attributes.
	lookup_cols = ["SA_GUID_2022",
				   "IHA_operational_name",
				   "HR_operational_name",]

	small_area_cols = ["SA_GUID_2022",
					   "SA_NUTS1_NAME",
					   "SA_NUTS2_NAME",
					   "SA_NUTS3_NAME",
					   "ED_ENGLISH",
					   "ED_GAEILGE",
					   "COUNTY_ENGLISH",
					   "COUNTY_GAEILGE",
					   "CSO_LEA",]

	require_columns(df=health_lookup,
					required=set(lookup_cols),
					name="health_lookup",)

	require_columns(df=small_areas,
					required=set(small_area_cols),
					name="small_areas",)

	health_lookup = health_lookup[lookup_cols].copy()
	small_areas_table = small_areas[small_area_cols].copy()

	ret = small_areas_table.merge(health_lookup,
								  on="SA_GUID_2022",
								  how="inner",
								  validate="one_to_one",)

	if ret.empty:
		raise ValueError("Geography merge produced no rows.")

	missing_hr = int(ret["HR_operational_name"].isna().sum())

	if missing_hr:
		raise ValueError(f"Geography merge has {missing_hr} rows without HR_operational_name.")

	return ret


def build_small_area_regions_geometry(small_areas: gpd.GeoDataFrame,
									  health_lookup: gpd.GeoDataFrame,
									  ) -> gpd.GeoDataFrame:
	#### Build small-area geometry with HSE health-region attributes for cached geometry outputs.
	lookup_cols = ["SA_GUID_2022",
				   "IHA_operational_name",
				   "HR_operational_name",]

	small_area_cols = ["SA_GUID_2022",
					   "SA_NUTS1_NAME",
					   "SA_NUTS2_NAME",
					   "SA_NUTS3_NAME",
					   "ED_ENGLISH",
					   "ED_GAEILGE",
					   "COUNTY_ENGLISH",
					   "COUNTY_GAEILGE",
					   "CSO_LEA",
					   "geometry",]

	require_columns(df=health_lookup,
					required=set(lookup_cols),
					name="health_lookup",)

	require_columns(df=small_areas,
					required=set(small_area_cols),
					name="small_areas",)

	health_lookup_table = health_lookup[lookup_cols].copy()
	small_areas_geometry = small_areas[small_area_cols].copy()

	ret = small_areas_geometry.merge(health_lookup_table,
									 on="SA_GUID_2022",
									 how="inner",
									 validate="one_to_one",)

	if ret.empty:
		raise ValueError("Small-area geometry merge produced no rows.")

	if ret["HR_operational_name"].isna().any():
		missing = int(ret["HR_operational_name"].isna().sum())
		raise ValueError(f"Small-area geometry merge has {missing} rows without HR_operational_name.")

	ret = gpd.GeoDataFrame(ret,
						   geometry="geometry",
						   crs=small_areas.crs,)

	ret["county_area"] = normalise_county_name(ret["COUNTY_ENGLISH"])
	ret["county_gaeilge"] = ret["COUNTY_GAEILGE"].astype("string").str.strip()
	ret["hse_region"] = normalise_region_name(ret["HR_operational_name"])

	return ret


def build_county_geometry(small_area_regions_geometry: gpd.GeoDataFrame,
						  paths: ProjectPaths,
						  ) -> gpd.GeoDataFrame:
	#### Build cached canonical county geometry using Gaeilge county identity.
	county_english_mapping = load_county_gaeilge_to_english_mapping(paths)

	required = {"county_gaeilge",
				"county_area",
				"SA_GUID_2022",
				"geometry",}

	require_columns(df=small_area_regions_geometry,
					required=required,
					name="small_area_regions_geometry",)

	work = small_area_regions_geometry.copy()
	work["county_english"] = work["county_gaeilge"].map(county_english_mapping)

	if work["county_english"].isna().any():
		missing_counties = sorted(work.loc[work["county_english"].isna(), "county_gaeilge"]
								  .dropna()
								  .unique()
								  .tolist())

		raise ValueError(f"Missing county Gaeilge to English mappings for county geometry: {missing_counties}")

	attributes = (work.groupby(["county_gaeilge", "county_english"],
							   as_index=False,
							   dropna=False,
							   observed=True,)
				  .agg(n_small_areas=("SA_GUID_2022", "nunique"),
					   n_county_areas=("county_area", "nunique"),
					   county_areas=("county_area", lambda x: "; ".join(sorted(x.dropna().unique()))),))

	ret = work.dissolve(by=["county_gaeilge", "county_english"],
						as_index=False,)

	ret = ret[["county_gaeilge", "county_english", "geometry"]].merge(attributes,
																	  on=["county_gaeilge", "county_english"],
																	  how="left",
																	  validate="one_to_one",)

	ret = (ret.sort_values("county_english")
		   .reset_index(drop=True))

	return ret


def build_county_area_geometry(small_area_regions_geometry: gpd.GeoDataFrame,
							   paths: ProjectPaths,
							   ) -> gpd.GeoDataFrame:
	#### Build cached county-area geometry retaining administrative county-area splits.
	county_english_mapping = load_county_gaeilge_to_english_mapping(paths)

	required = {"county_gaeilge",
				"county_area",
				"SA_GUID_2022",
				"geometry",}

	require_columns(df=small_area_regions_geometry,
					required=required,
					name="small_area_regions_geometry",)

	work = small_area_regions_geometry.copy()
	work["county_english"] = work["county_gaeilge"].map(county_english_mapping)

	if work["county_english"].isna().any():
		missing_counties = sorted(work.loc[work["county_english"].isna(), "county_gaeilge"]
								  .dropna()
								  .unique()
								  .tolist())

		raise ValueError(f"Missing county Gaeilge to English mappings for county-area geometry: {missing_counties}")

	attributes = (work.groupby(["county_gaeilge",
								"county_english",
								"county_area",],
							   as_index=False,
							   dropna=False,
							   observed=True,)
				  .agg(n_small_areas=("SA_GUID_2022", "nunique"),))

	ret = work.dissolve(by=["county_gaeilge", "county_english", "county_area"],
						as_index=False,)

	ret = ret[["county_gaeilge", "county_english", "county_area", "geometry"]].merge(
		attributes,
		on=["county_gaeilge", "county_english", "county_area"],
		how="left",
		validate="one_to_one",
	)

	ret = (ret.sort_values(["county_english", "county_area"])
		   .reset_index(drop=True))

	return ret


def build_hse_region_geometry(small_area_regions_geometry: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
	#### Build cached HSE-region geometry from small-area geometries.
	required = {"hse_region",
				"county_area",
				"county_gaeilge",
				"SA_GUID_2022",
				"geometry",}

	require_columns(df=small_area_regions_geometry,
					required=required,
					name="small_area_regions_geometry",)

	work = small_area_regions_geometry.copy()

	attributes = (work.groupby("hse_region",
							   as_index=False,
							   dropna=False,
							   observed=True,)
				  .agg(n_small_areas=("SA_GUID_2022", "nunique"),
					   n_county_areas=("county_area", "nunique"),
					   n_canonical_counties=("county_gaeilge", "nunique"),
					   county_areas=("county_area", lambda x: "; ".join(sorted(x.dropna().unique()))),))

	ret = work.dissolve(by="hse_region",
						as_index=False,)

	ret = ret[["hse_region", "geometry"]].merge(attributes,
												on="hse_region",
												how="left",
												validate="one_to_one",)

	ret = (ret.sort_values("hse_region")
		   .reset_index(drop=True))

	return ret


def write_cached_geometry_outputs(dirs: dict[str, Path],
								  logger,
								  small_areas: gpd.GeoDataFrame,
								  health_lookup: gpd.GeoDataFrame,
								  paths: ProjectPaths,
								  ) -> dict[str, Path]:
	#### Build and write cached geometry outputs for EDA mapping.
	small_area_regions_geometry = build_small_area_regions_geometry(small_areas=small_areas,
																	health_lookup=health_lookup,)

	county_geometry = build_county_geometry(small_area_regions_geometry=small_area_regions_geometry,
											paths=paths,)

	county_area_geometry = build_county_area_geometry(small_area_regions_geometry=small_area_regions_geometry,
													  paths=paths,)

	hse_region_geometry = build_hse_region_geometry(small_area_regions_geometry)

	output_paths = {"geo_hse_region_geometry": dirs["processed"] / "geo_hse_region_geometry.gpkg",
					"geo_county_geometry": dirs["processed"] / "geo_county_geometry.gpkg",
					"geo_county_area_geometry": dirs["processed"] / "geo_county_area_geometry.gpkg",}

	hse_region_geometry.to_file(output_paths["geo_hse_region_geometry"],
								driver="GPKG",
								layer="geo_hse_region_geometry",)

	county_geometry.to_file(output_paths["geo_county_geometry"],
							driver="GPKG",
							layer="geo_county_geometry",)

	county_area_geometry.to_file(output_paths["geo_county_area_geometry"],
								 driver="GPKG",
								 layer="geo_county_area_geometry",)

	write_column_audit(hse_region_geometry.drop(columns="geometry", errors="ignore"),
					   dirs["manifests"] / "geo_hse_region_geometry_column_audit.csv",)

	write_column_audit(county_geometry.drop(columns="geometry", errors="ignore"),
					   dirs["manifests"] / "geo_county_geometry_column_audit.csv",)

	write_column_audit(county_area_geometry.drop(columns="geometry", errors="ignore"),
					   dirs["manifests"] / "geo_county_area_geometry_column_audit.csv",)

	log_block(logger,
			  "Cached GEO geometry outputs saved",
			  {"geo_hse_region_geometry": str(output_paths["geo_hse_region_geometry"]),
			   "geo_hse_region_rows": int(hse_region_geometry.shape[0]),
			   "geo_county_geometry": str(output_paths["geo_county_geometry"]),
			   "geo_county_rows": int(county_geometry.shape[0]),
			   "geo_county_area_geometry": str(output_paths["geo_county_area_geometry"]),
			   "geo_county_area_rows": int(county_area_geometry.shape[0]),
			   "crs": str(small_area_regions_geometry.crs),},)

	return output_paths


def run_geo_preprocessing(paths: ProjectPaths) -> dict[str, Path]:
	#### Build processed HSE health region geography and county-region baseline mapping.
	stage = paths.stage_from_module(__name__)
	dirs = paths.stage_dirs(stage)

	ensure_directories(dirs)

	logger = get_logger(name="preprocess_geo",
						log_dir=dirs["logs"],
						filename="preprocess_geo.log",)

	logger.info("Starting GEO preprocessing")
	logger.info("Loading geographic datasets")

	small_areas, health_lookup = load_geo_inputs(paths)

	write_geo_audits(dirs=dirs,
					 logger=logger,
					 small_areas=small_areas,
					 health_lookup=health_lookup,)

	logger.info("Building small area county/town lookup")
	small_area_lookup = build_small_area_county_towns_lookup(small_areas)

	lookup_output_path = save_dataset_with_manifest(df=small_area_lookup,
													output_path=dirs["processed"] / "geo_small_areas_county_towns.csv",
													manifest_dir=dirs["manifests"],
													stage="geo_small_areas_county_towns",)


	log_block(logger,
			  "Small area county/town lookup saved",
			  {"csv_path": str(lookup_output_path),
			   "rows": int(small_area_lookup.shape[0]),},)

	logger.info("Building health region table")
	small_area_regions = build_health_regions(small_areas=small_areas,
											  health_lookup=health_lookup,)

	health_regions_output_path = save_dataset_with_manifest(df=small_area_regions,
															output_path=dirs["processed"] / "geo_health_regions.csv",
															manifest_dir=dirs["manifests"],
															stage="geo_health_regions",)

	log_block(logger,
			  "Processed GEO table saved",
			  {"path": str(health_regions_output_path),
			   "rows": int(small_area_regions.shape[0]),},)

	logger.info("Building cached geometry outputs")
	geometry_output_paths = write_cached_geometry_outputs(dirs=dirs,
														  logger=logger,
														  small_areas=small_areas,
														  health_lookup=health_lookup,
														  paths=paths,)

	logger.info("Building GeoHive hospital geography and capability outputs")
	hospital_output_paths = write_hospital_geo_outputs(dirs=dirs,
													   logger=logger,
													   small_areas=small_areas,
													   health_lookup=health_lookup,
													   paths=paths,)

	logger.info("Building county-to-region baseline mapping")
	county_region = build_county_to_region_baseline(small_area_regions=small_area_regions,
													small_areas=small_areas,
													paths=paths,)

	validate_county_to_region_baseline(county_region)

	canonical_split_counties = (county_region.loc[county_region["n_hse_regions"].gt(1),
												  ["county_gaeilge",
												   "county_english",
												   "county_area",
												   "hse_region",]]
								.drop_duplicates()
								.sort_values(["county_gaeilge",
											  "county_english",
											  "county_area",
											  "hse_region",]))

	multi_area_counties = (county_region.loc[county_region["n_county_areas"].gt(1),
											 ["county_gaeilge",
											  "county_english",
											  "county_area",]]
						   .drop_duplicates()
						   .sort_values(["county_gaeilge",
										 "county_english",
										 "county_area",]))

	log_block(logger,
			  "County-region baseline weighting method",
			  {"canonical_county_field": "county_gaeilge",
			   "weight_numerator": "n_small_areas in county_gaeilge-county_area-hse_region group",
			   "weight_denominator": "total_small_areas across county_gaeilge",
			   "weight_column": "small_area_weight",
			   "reason": "COUNTY_ENGLISH may contain administrative splits such as North/South Tipperary; COUNTY_GAEILGE is used as the canonical county identity.",
			   "multi_area_counties": multi_area_counties.to_dict(orient="records"),
			   "split_counties": canonical_split_counties.to_dict(orient="records"),},)

	county_region_output_path = save_dataset_with_manifest(df=county_region,
														   output_path=dirs["processed"] / "geo_county_to_region_baseline.csv",
														   manifest_dir=dirs["manifests"],
														   stage="geo_county_to_region_baseline",)

	split_counties = (county_region.loc[county_region["n_hse_regions"].gt(1), "county_gaeilge"]
					  .drop_duplicates()
					  .to_list())

	log_block(logger,
			  "County-region baseline saved",
			  {"path": str(county_region_output_path),
			   "rows": int(county_region.shape[0]),
			   "split_counties": split_counties,},)

	logger.info("Building Census 2022 sex-count outputs from Table 1.2 small-area data")
	census_sex_output_paths = run_census_2022_sex_preprocessing(small_area_regions=small_area_regions,
																  paths=paths,
																  dirs=dirs,
																  logger=logger,)

	outputs = {"geo_small_areas_county_towns": lookup_output_path,
			   "geo_health_regions": health_regions_output_path,
			   "geo_county_to_region_baseline": county_region_output_path,
			   **geometry_output_paths,
			   **hospital_output_paths,
			   **census_sex_output_paths,}

	log_block(logger,
			  "GEO preprocessing outputs saved",
			  {key: str(value) for key, value in outputs.items()},)

	logger.info("Finished GEO preprocessing")

	return outputs