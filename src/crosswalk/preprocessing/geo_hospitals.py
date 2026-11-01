#### geo_hospitals.py

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

from crosswalk.paths import ProjectPaths
from crosswalk.preprocessing.geo_common import (
	build_county_match_lookup,
	load_geo_input_schema,
	load_hospital_model_group_mapping,
	normalise_county_name,
	normalise_region_name,
	normalise_region_name_with_reference,
)
from crosswalk.utils.io import save_dataset_with_manifest
from crosswalk.utils.logging import log_block
from crosswalk.utils.manifest import write_column_audit
from crosswalk.utils.schema import apply_schema
from crosswalk.utils.text import (
	extract_last_filled_text_field,
	fuzzy_match_to_candidates,
	normalise_county_candidate_label,
	normalise_lookup_label,
)
from crosswalk.utils.validation import (
	require_columns,
	validate_no_missing,
)


HOSPITAL_FILE = "Hospitals_-_HSE_Ireland.geojson"

HOSPITAL_MODEL_SOURCE_DIR = "esri_2025"
HOSPITAL_MODEL_FILE = "hse_hospital_model_numbers_2023.csv"

GEO_HOSPITALS_FILE = "geo_hospitals.csv"
GEO_HOSPITALS_GEOMETRY_FILE = "geo_hospitals_geometry.gpkg"
GEO_HOSPITAL_COUNTY_MATCH_AUDIT_FILE = "geo_hospital_county_match_audit.csv"
GEO_HOSPITAL_MODEL_MATCH_AUDIT_FILE = "geo_public_acute_hospital_match_audit.csv"
GEO_HOSPITAL_CAPABILITY_SUMMARY_BY_COUNTY_FILE = "geo_hospital_capability_summary_by_county.csv"
GEO_HOSPITAL_CAPABILITY_SUMMARY_BY_HSE_REGION_FILE = "geo_hospital_capability_summary_by_hse_region.csv"

GEO_PUBLIC_ACUTE_HOSPITAL_POINTS_FILE = "geo_public_acute_hospital_points_2023.csv"
GEO_PUBLIC_ACUTE_HOSPITAL_POINTS_GEOMETRY_FILE = "geo_public_acute_hospital_points_2023.gpkg"
GEO_PUBLIC_ACUTE_HOSPITAL_MATCH_AUDIT_FILE = "geo_public_acute_hospital_match_audit.csv"


def load_hospital_inputs(paths: ProjectPaths) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
	#### Load raw GeoHive hospital points and published hospital model classification data.
	hospital_path = paths.raw / HOSPITAL_FILE
	hospital_model_path = paths.published / HOSPITAL_MODEL_SOURCE_DIR / HOSPITAL_MODEL_FILE

	for input_path in [hospital_path, hospital_model_path]:
		if not input_path.exists():
			raise FileNotFoundError(f"Required hospital geography file not found: {input_path}")

	hospitals = gpd.read_file(hospital_path)

	hospital_models = pd.read_csv(hospital_model_path)
	hospital_models = apply_schema(hospital_models,
								   load_geo_input_schema(paths=paths,
														 schema_name="hospital_models"),)

	return hospitals, hospital_models


def write_hospital_raw_audits(dirs: dict[str, Path],
							  logger,
							  hospitals: gpd.GeoDataFrame,
							  hospital_models: pd.DataFrame,
							  ) -> None:
	#### Write raw hospital input column audits.
	hospitals_audit_path = dirs["manifests"] / "geo_hospitals_raw_column_audit.csv"
	hospital_models_audit_path = dirs["manifests"] / "geo_hospital_models_raw_column_audit.csv"

	write_column_audit(hospitals.drop(columns="geometry", errors="ignore"),
					   hospitals_audit_path,)

	write_column_audit(hospital_models,
					   hospital_models_audit_path,)

	log_block(logger,
			  "Raw hospital column audits saved",
			  {"hospitals": str(hospitals_audit_path),
			   "hospital_models": str(hospital_models_audit_path),},)


def build_geohive_hospital_geographies(hospitals: gpd.GeoDataFrame,
                                       small_areas: gpd.GeoDataFrame,
                                       health_lookup: gpd.GeoDataFrame,
                                       paths: ProjectPaths,
                                       ) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
	#### Build GeoHive hospital geography output without public acute model classification.
	county_matched = attach_hospital_reported_county(hospitals=hospitals,
	                                                 paths=paths, )

	ret = attach_hospital_spatial_geography(hospitals=county_matched,
	                                        small_areas=small_areas,
	                                        health_lookup=health_lookup, )

	ret["hospital_lookup_key"] = normalise_lookup_label(ret["hospital_name"])

	ret_cols = [
		"hospital_id",
		"hospital_category",
		"hospital_subcategory",
		"hospital_name",
		"hospital_lookup_key",
		"hospital_address",
		"eircode",
		"point_x",
		"point_y",
		"reported_county_raw",
		"reported_county_clean",
		"county_gaeilge",
		"county_english",
		"county_match_score",
		"county_match_method",
		"county_match_review_required",
		"address_county_matches_spatial_county",
		"SA_GUID_2022",
		"spatial_county_area",
		"spatial_county_gaeilge",
		"spatial_nuts3_name",
		"spatial_lea_name",
		"iha_operational_name",
		"hse_region",
		"geometry",
	]

	missing_cols = [column for column in ret_cols if column not in ret.columns]

	if missing_cols:
		raise ValueError(
			"GeoHive hospital geography build is missing expected columns: "
			f"{missing_cols}"
		)

	ret = ret[ret_cols].copy()

	ret = (ret.sort_values(["hse_region",
	                        "county_english",
	                        "hospital_name", ])
	       .reset_index(drop=True))

	ret = gpd.GeoDataFrame(ret,
	                       geometry="geometry",
	                       crs=county_matched.crs, )

	county_match_audit_cols = [
		"hospital_id",
		"hospital_name",
		"hospital_address",
		"reported_county_raw",
		"reported_county_clean",
		"county_gaeilge",
		"county_english",
		"county_match_score",
		"county_match_method",
		"county_match_review_required",
		"spatial_county_area",
		"spatial_county_gaeilge",
		"spatial_nuts3_name",
		"hse_region",
		"address_county_matches_spatial_county",
	]

	county_match_audit = pd.DataFrame(
		ret.drop(columns="geometry", errors="ignore")
	)[county_match_audit_cols].copy()

	return ret, county_match_audit


def build_public_acute_hospital_points(hospital_models: pd.DataFrame,
                                       geohive_hospitals: gpd.GeoDataFrame,
                                       paths: ProjectPaths,
                                       min_match_score: float = 82.0,
                                       ) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
	#### Match authoritative public acute hospital model rows to GeoHive hospital point locations.
	models = validate_hospital_models(hospital_models=hospital_models,
	                                  paths=paths, )

	models = models.reset_index(drop=True).copy()
	models["public_acute_hospital_id"] = models.index + 1

	geohive = geohive_hospitals.copy()

	geohive["geohive_hospital_lookup_key"] = normalise_lookup_label(geohive["hospital_name"])
	geohive["geohive_county_clean"] = normalise_county_candidate_label(geohive["county_english"])
	geohive["geohive_hse_region"] = normalise_region_name_with_reference(geohive["hse_region"],
	                                                                     paths=paths, )

	geohive_lookup_cols = [
		"hospital_id",
		"hospital_name",
		"hospital_lookup_key",
		"geohive_hospital_lookup_key",
		"hospital_address",
		"eircode",
		"county_english",
		"geohive_county_clean",
		"hse_region",
		"geohive_hse_region",
		"spatial_county_area",
		"spatial_nuts3_name",
		"spatial_lea_name",
		"iha_operational_name",
		"point_x",
		"point_y",
		"geometry",
	]

	geohive_lookup = geohive[geohive_lookup_cols].copy()

	geohive_lookup = geohive_lookup.rename(columns={
		"hospital_id": "geohive_hospital_id",
		"hospital_name": "geohive_hospital_name",
		"hospital_lookup_key": "geohive_original_hospital_lookup_key",
		"hospital_address": "geohive_hospital_address",
		"eircode": "geohive_eircode",
		"county_english": "geohive_county_english",
		"hse_region": "geohive_spatial_hse_region",
		"spatial_county_area": "geohive_spatial_county_area",
		"spatial_nuts3_name": "geohive_spatial_nuts3_name",
		"spatial_lea_name": "geohive_spatial_lea_name",
		"iha_operational_name": "geohive_iha_operational_name",
		"point_x": "geohive_point_x",
		"point_y": "geohive_point_y",
	})

	geohive_lookup = (geohive_lookup.sort_values(["geohive_county_clean",
	                                              "geohive_hospital_name",
	                                              "geohive_hospital_id", ])
	                  .drop_duplicates("geohive_hospital_lookup_key",
	                                   keep="first"))

	match_rows = []

	for _, model_row in models.iterrows():
		county_candidates = geohive_lookup.loc[
			geohive_lookup["geohive_county_clean"].eq(model_row["model_county_clean"])
		].copy()

		if county_candidates.empty:
			candidates = geohive_lookup["geohive_hospital_lookup_key"].dropna().to_list()
			match_scope = "all_geohive_hospitals_fallback"
		else:
			candidates = county_candidates["geohive_hospital_lookup_key"].dropna().to_list()
			match_scope = "same_county_geohive_hospitals"

		match_key, match_score = fuzzy_match_to_candidates(value=model_row["model_hospital_lookup_key"],
		                                                   candidates=candidates, )

		match_rows.append({
			"public_acute_hospital_id": model_row["public_acute_hospital_id"],
			"geohive_hospital_lookup_key": match_key,
			"public_acute_match_score": match_score,
			"public_acute_match_scope": match_scope,
		})

	matches = pd.DataFrame(match_rows)

	ret = models.merge(matches,
	                   on="public_acute_hospital_id",
	                   how="left",
	                   validate="one_to_one", )

	ret = ret.merge(geohive_lookup,
	                on="geohive_hospital_lookup_key",
	                how="left",
	                validate="many_to_one", )

	ret["model_county_matches_geohive_county"] = ret["model_county_clean"].eq(
		ret["geohive_county_clean"]
	)

	ret["model_hse_region_matches_geohive_spatial_hse_region"] = ret["model_hse_region"].eq(
		ret["geohive_hse_region"]
	)

	ret["public_acute_match_accepted"] = (
			ret["public_acute_match_score"].ge(min_match_score)
			& ret["model_county_matches_geohive_county"].fillna(False)
	)

	ret["public_acute_match_review_required"] = ~ret["public_acute_match_accepted"]

	ret.loc[~ret["public_acute_match_accepted"], [
		"geohive_hospital_id",
		"geohive_hospital_name",
		"geohive_hospital_address",
		"geohive_eircode",
		"geohive_point_x",
		"geohive_point_y",
		"geometry",
	]] = pd.NA

	ret = gpd.GeoDataFrame(ret,
	                       geometry="geometry",
	                       crs=geohive_hospitals.crs, )

	audit_cols = [
		"public_acute_hospital_id",
		"model_hospital_name",
		"model_hse_region",
		"model_county_raw",
		"model_county_clean",
		"hospital_model_number",
		"hospital_model_group",
		"specialist_type",
		"geohive_hospital_name",
		"geohive_county_english",
		"geohive_spatial_hse_region",
		"public_acute_match_score",
		"public_acute_match_scope",
		"model_county_matches_geohive_county",
		"model_hse_region_matches_geohive_spatial_hse_region",
		"public_acute_match_accepted",
		"public_acute_match_review_required",
	]

	match_audit = pd.DataFrame(
		ret.drop(columns="geometry", errors="ignore")
	)[audit_cols].copy()

	return ret, match_audit


def build_hospital_geographies(hospitals: gpd.GeoDataFrame,
                               hospital_models: pd.DataFrame,
                               small_areas: gpd.GeoDataFrame,
                               health_lookup: gpd.GeoDataFrame,
                               paths: ProjectPaths,
                               ) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
	#### Build processed hospital geography output with spatial geography, county audit fields and accepted model classifications.
	county_matched = attach_hospital_reported_county(hospitals=hospitals,
	                                                 paths=paths, )

	spatially_joined = attach_hospital_spatial_geography(hospitals=county_matched,
	                                                     small_areas=small_areas,
	                                                     health_lookup=health_lookup, )

	modelled, model_match_audit = attach_hospital_model_classification(
		hospitals=spatially_joined,
		hospital_models=hospital_models,
		paths=paths,
	)

	ret_cols = [
		"hospital_id",
		"hospital_category",
		"hospital_subcategory",
		"hospital_name",
		"hospital_address",
		"eircode",
		"point_x",
		"point_y",
		"reported_county_raw",
		"reported_county_clean",
		"county_gaeilge",
		"county_english",
		"county_match_score",
		"county_match_method",
		"county_match_review_required",
		"address_county_matches_spatial_county",
		"SA_GUID_2022",
		"spatial_county_area",
		"spatial_county_gaeilge",
		"spatial_nuts3_name",
		"spatial_lea_name",
		"iha_operational_name",
		"hse_region",
		"hospital_lookup_key",
		"model_hospital_lookup_key",
		"model_hospital_name",
		"model_hse_region",
		"model_county_raw",
		"model_county_clean",
		"model_county_matches_geo_county",
		"model_hse_region_matches_spatial_hse_region",
		"hospital_model_number",
		"hospital_model_group",
		"is_model_4",
		"is_model_3",
		"is_model_2",
		"is_model_3_or_4",
		"is_model_2_to_4",
		"is_specialist_hospital",
		"specialist_type",
		"hospital_model_source",
		"hospital_model_match_score",
		"hospital_model_match_score_accepted",
		"hospital_model_match_method",
		"hospital_model_match_accepted",
		"hospital_model_match_review_required",
		"geometry",
	]

	missing_cols = [column for column in ret_cols if column not in modelled.columns]

	if missing_cols:
		raise ValueError(
			"Hospital geography build is missing expected columns after model matching: "
			f"{missing_cols}"
		)

	ret = modelled[ret_cols].copy()

	ret = (ret.sort_values(["hse_region",
	                        "county_english",
	                        "hospital_name", ])
	       .reset_index(drop=True))

	ret = gpd.GeoDataFrame(ret,
	                       geometry="geometry",
	                       crs=modelled.crs, )

	county_match_audit_cols = [
		"hospital_id",
		"hospital_name",
		"hospital_address",
		"reported_county_raw",
		"reported_county_clean",
		"county_gaeilge",
		"county_english",
		"county_match_score",
		"county_match_method",
		"county_match_review_required",
		"spatial_county_area",
		"spatial_county_gaeilge",
		"spatial_nuts3_name",
		"hse_region",
		"address_county_matches_spatial_county",
	]

	county_match_audit = pd.DataFrame(
		ret.drop(columns="geometry", errors="ignore")
	)[county_match_audit_cols].copy()

	return ret, county_match_audit, model_match_audit


def summarise_hospital_capability_by_county(hospitals: gpd.GeoDataFrame) -> pd.DataFrame:
	#### Summarise hospital model capability by canonical county.
	df = hospitals.copy()

	for column in ["is_model_4",
	               "is_model_3",
	               "is_model_2",
	               "is_model_3_or_4",
	               "is_model_2_to_4",
	               "is_specialist_hospital", ]:
		df[column] = df[column].fillna(False).astype(bool)

	ret = (df.groupby(["county_gaeilge", "county_english"],
	                  as_index=False,
	                  dropna=False,
	                  observed=True, )
	       .agg(n_hospitals=("hospital_id", "nunique"),
	            n_model_4_hospitals=("is_model_4", "sum"),
	            n_model_3_hospitals=("is_model_3", "sum"),
	            n_model_2_hospitals=("is_model_2", "sum"),
	            n_model_3_or_4_hospitals=("is_model_3_or_4", "sum"),
	            n_model_2_to_4_hospitals=("is_model_2_to_4", "sum"),
	            n_specialist_hospitals=("is_specialist_hospital", "sum"), )
	       .sort_values("county_english")
	       .reset_index(drop=True))

	return ret


def join_names_where(df: pd.DataFrame,
                     index,
                     flag_column: str,
                     name_column: str = "hospital_name",
                     ) -> str:
	#### Join hospital names within a group where a Boolean flag is true.
	group = df.loc[index].copy()
	ret = "; ".join(sorted(group.loc[group[flag_column].fillna(False),
	name_column].dropna().astype(str).unique()))

	return ret


def summarise_hospital_capability_by_hse_region(hospitals: gpd.GeoDataFrame) -> pd.DataFrame:
	#### Summarise hospital model capability by HSE region.
	df = hospitals.copy()

	for column in ["is_model_4",
	               "is_model_3",
	               "is_model_2",
	               "is_model_3_or_4",
	               "is_model_2_to_4",
	               "is_specialist_hospital", ]:
		df[column] = df[column].fillna(False).astype(bool)

	ret = (df.groupby("hse_region",
	                  as_index=False,
	                  dropna=False,
	                  observed=True, )
	       .agg(n_hospitals=("hospital_id", "nunique"),
	            n_model_4_hospitals=("is_model_4", "sum"),
	            n_model_3_hospitals=("is_model_3", "sum"),
	            n_model_2_hospitals=("is_model_2", "sum"),
	            n_model_3_or_4_hospitals=("is_model_3_or_4", "sum"),
	            n_model_2_to_4_hospitals=("is_model_2_to_4", "sum"),
	            n_specialist_hospitals=("is_specialist_hospital", "sum"), ))

	name_rows = []

	for hse_region, group in df.groupby("hse_region", dropna=False, observed=True):
		name_rows.append({
			"hse_region": hse_region,
			"model_4_hospital_names": join_names_where(df, group.index, "is_model_4"),
			"model_3_or_4_hospital_names": join_names_where(df, group.index, "is_model_3_or_4"),
			"specialist_hospital_names": join_names_where(df, group.index, "is_specialist_hospital"),
		})

	names = pd.DataFrame(name_rows)

	ret = (ret.merge(names,
	                 on="hse_region",
	                 how="left",
	                 validate="one_to_one")
	       .sort_values("hse_region")
	       .reset_index(drop=True))

	return ret


def build_small_area_regions_geometry(small_areas: gpd.GeoDataFrame,
									  health_lookup: gpd.GeoDataFrame,
									  ) -> gpd.GeoDataFrame:
	#### Build small-area geometry with HSE health-region attributes for hospital spatial joins.
	lookup_cols = ["SA_GUID_2022",
				   "IHA_operational_name",
				   "HR_operational_name",]

	small_area_cols = ["SA_GUID_2022",
					   "SA_NUTS3_NAME",
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
		raise ValueError("Small-area geometry merge produced no rows for hospital spatial join.")

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


def validate_geohive_hospitals(hospitals: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
	#### Validate and standardise GeoHive hospital point records.
	required = {
		"OBJECTID",
		"category",
		"subcategory",
		"name",
		"address1",
		"address2",
		"address3",
		"address4",
		"Eircode",
		"POINT_X",
		"POINT_Y",
		"geometry",
	}

	require_columns(df=hospitals,
					required=required,
					name="geohive_hospitals",)

	ret = hospitals.copy()

	ret["hospital_id"] = pd.to_numeric(ret["OBJECTID"], errors="coerce").astype("Int64")
	ret["hospital_category"] = ret["category"].astype("string").str.strip()
	ret["hospital_subcategory"] = ret["subcategory"].astype("string").str.strip()
	ret["hospital_name"] = ret["name"].astype("string").str.strip()
	ret["eircode"] = ret["Eircode"].astype("string").str.strip()

	for column in ["address1", "address2", "address3", "address4"]:
		ret[column] = ret[column].astype("string").str.strip()

	ret["point_x"] = pd.to_numeric(ret["POINT_X"], errors="coerce")
	ret["point_y"] = pd.to_numeric(ret["POINT_Y"], errors="coerce")

	ret["hospital_address"] = (
		ret[["address1", "address2", "address3", "address4"]]
		.fillna("")
		.agg(lambda row: ", ".join([value for value in row if str(value).strip() != ""]), axis=1)
		.astype("string")
	)

	ret["reported_county_raw"] = extract_last_filled_text_field(df=ret,
																columns=["address1",
																		 "address2",
																		 "address3",
																		 "address4",],)

	ret["reported_county_clean"] = normalise_county_candidate_label(ret["reported_county_raw"])

	validate_no_missing(df=ret,
						columns=["hospital_id", "hospital_name", "reported_county_clean"],
						name="geohive_hospitals",)

	if ret.crs is None:
		ret = ret.set_crs("EPSG:4326")

	ret = gpd.GeoDataFrame(ret,
						   geometry="geometry",
						   crs=ret.crs,)

	return ret


def attach_hospital_reported_county(hospitals: gpd.GeoDataFrame,
									paths: ProjectPaths,
									min_match_score: float = 85.0,
									) -> gpd.GeoDataFrame:
	#### Attach canonical county labels using reported county parsed from hospital address fields.
	hospital_points = validate_geohive_hospitals(hospitals)
	county_lookup = build_county_match_lookup(paths)

	candidates = (county_lookup["county_lookup_key"]
				  .dropna()
				  .drop_duplicates()
				  .sort_values()
				  .to_list())

	matches = hospital_points["reported_county_clean"].apply(
		lambda value: fuzzy_match_to_candidates(value=value,
												candidates=candidates,)
	)

	hospital_points["county_lookup_key"] = matches.map(lambda item: item[0])
	hospital_points["county_match_score"] = matches.map(lambda item: item[1])
	hospital_points["county_match_method"] = "last_filled_address_field_fuzzy_match"
	hospital_points["county_match_review_required"] = hospital_points["county_match_score"].lt(min_match_score)

	hospital_points = hospital_points.merge(county_lookup,
											on="county_lookup_key",
											how="left",
											validate="many_to_one",)

	if hospital_points["county_english"].isna().any():
		missing = hospital_points.loc[hospital_points["county_english"].isna(),
									  ["hospital_id",
									   "hospital_name",
									   "reported_county_raw",
									   "reported_county_clean",
									   "county_lookup_key",
									   "county_match_score",]]

		raise ValueError(f"Unable to map hospital counties: {missing.to_dict(orient='records')}")

	return hospital_points


def validate_hospital_models(hospital_models: pd.DataFrame,
							 paths: ProjectPaths,
							 ) -> pd.DataFrame:
	#### Validate and standardise hospital model classification data.
	required = {
		"source",
		"hse_health_region",
		"hospital_name",
		"county",
		"hospital_model_number",
		"specialist",
	}

	require_columns(df=hospital_models,
					required=required,
					name="hospital_models",)

	model_group_mapping = load_hospital_model_group_mapping(paths)

	ret = hospital_models.copy()

	ret["hospital_model_source"] = ret["source"].astype("string").str.strip()
	ret["model_hse_region"] = normalise_region_name_with_reference(ret["hse_health_region"],
																   paths=paths,)
	ret["model_hospital_name"] = ret["hospital_name"].astype("string").str.strip()
	ret["model_county_raw"] = ret["county"].astype("string").str.strip()
	ret["hospital_model_number"] = pd.to_numeric(ret["hospital_model_number"],
												 errors="coerce").astype("Int64")
	ret["specialist_type"] = (ret["specialist"]
							  .astype("string")
							  .str.strip()
							  .replace("", pd.NA))

	ret["is_specialist_hospital"] = ret["specialist_type"].notna()
	ret["is_model_4"] = ret["hospital_model_number"].eq(4)
	ret["is_model_3"] = ret["hospital_model_number"].eq(3)
	ret["is_model_2"] = ret["hospital_model_number"].eq(2)
	ret["is_model_3_or_4"] = ret["hospital_model_number"].isin([3, 4])
	ret["is_model_2_to_4"] = ret["hospital_model_number"].isin([2, 3, 4])

	ret["hospital_model_group"] = ret["hospital_model_number"].astype("string").map(model_group_mapping)
	ret.loc[ret["hospital_model_number"].isna() & ret["is_specialist_hospital"], "hospital_model_group"] = (
		model_group_mapping.get("specialist", "specialist")
	)

	ret["model_county_clean"] = normalise_county_candidate_label(ret["model_county_raw"])
	ret["model_hospital_lookup_key"] = normalise_lookup_label(ret["model_hospital_name"])

	validate_no_missing(df=ret,
						columns=["model_hospital_name", "model_hse_region", "model_county_clean"],
						name="hospital_models",)

	return ret


def attach_hospital_spatial_geography(hospitals: gpd.GeoDataFrame,
									  small_areas: gpd.GeoDataFrame,
									  health_lookup: gpd.GeoDataFrame,
									  ) -> gpd.GeoDataFrame:
	#### Attach small-area, NUTS3 and HSE-region geography using hospital point location.
	small_area_regions_geometry = build_small_area_regions_geometry(small_areas=small_areas,
																	health_lookup=health_lookup,)

	hospital_points = hospitals.copy()

	if hospital_points.crs != small_area_regions_geometry.crs:
		hospital_points = hospital_points.to_crs(small_area_regions_geometry.crs)

	spatial_cols = [
		"SA_GUID_2022",
		"SA_NUTS3_NAME",
		"COUNTY_ENGLISH",
		"COUNTY_GAEILGE",
		"CSO_LEA",
		"IHA_operational_name",
		"HR_operational_name",
		"geometry",
	]

	ret = gpd.sjoin(hospital_points,
					small_area_regions_geometry[spatial_cols],
					how="left",
					predicate="within",)

	ret = ret.drop(columns=["index_right"], errors="ignore")

	if ret["SA_GUID_2022"].isna().any():
		missing = ret.loc[ret["SA_GUID_2022"].isna(),
						  ["hospital_id", "hospital_name", "hospital_address"]]

		raise ValueError(f"Some hospitals could not be spatially joined: {missing.to_dict(orient='records')}")

	ret["spatial_county_area"] = normalise_county_name(ret["COUNTY_ENGLISH"])
	ret["spatial_county_gaeilge"] = ret["COUNTY_GAEILGE"].astype("string").str.strip()
	ret["spatial_nuts3_name"] = ret["SA_NUTS3_NAME"].astype("string").str.strip()
	ret["spatial_lea_name"] = ret["CSO_LEA"].astype("string").str.strip()
	ret["iha_operational_name"] = ret["IHA_operational_name"].astype("string").str.strip()
	ret["hse_region"] = normalise_region_name(ret["HR_operational_name"])

	ret["address_county_matches_spatial_county"] = (
		ret["county_gaeilge"].astype("string").eq(ret["spatial_county_gaeilge"].astype("string"))
	)

	return ret


def attach_hospital_model_classification(hospitals: gpd.GeoDataFrame,
										 hospital_models: pd.DataFrame,
										 paths: ProjectPaths,
										 min_match_score: float = 82.0,
										 require_county_match: bool = True,
										 ) -> tuple[gpd.GeoDataFrame, pd.DataFrame]:
	#### Attach hospital model classifications to GeoHive hospital records using conservative fuzzy matching.
	models = validate_hospital_models(hospital_models,
									  paths=paths,)
	ret = hospitals.copy()

	model_candidates = (
		models["model_hospital_lookup_key"]
		.dropna()
		.drop_duplicates()
		.sort_values()
		.to_list()
	)

	ret["hospital_lookup_key"] = normalise_lookup_label(ret["hospital_name"])

	matches = ret["hospital_lookup_key"].apply(
		lambda value: fuzzy_match_to_candidates(value=value,
												candidates=model_candidates,)
	)

	ret["model_hospital_lookup_key"] = matches.map(lambda item: item[0])
	ret["hospital_model_match_score"] = matches.map(lambda item: item[1])

	model_cols = [
		"model_hospital_lookup_key",
		"model_hospital_name",
		"model_hse_region",
		"model_county_raw",
		"model_county_clean",
		"hospital_model_number",
		"hospital_model_group",
		"is_model_4",
		"is_model_3",
		"is_model_2",
		"is_model_3_or_4",
		"is_model_2_to_4",
		"is_specialist_hospital",
		"specialist_type",
		"hospital_model_source",
	]

	ret = ret.merge(models[model_cols],
					on="model_hospital_lookup_key",
					how="left",
					validate="many_to_one",)

	ret["hospital_model_match_method"] = "fuzzy_hospital_name_to_esri_2025_appendix_a"

	ret["model_county_matches_geo_county"] = ret["model_county_clean"].eq(
		normalise_county_candidate_label(ret["county_english"])
	)

	ret["model_hse_region_matches_spatial_hse_region"] = ret["model_hse_region"].eq(
		normalise_region_name_with_reference(ret["hse_region"],
											 paths=paths,)
	)

	ret["hospital_model_match_score_accepted"] = ret["hospital_model_match_score"].ge(min_match_score)

	if require_county_match:
		ret["hospital_model_match_accepted"] = (
				ret["hospital_model_match_score_accepted"]
				& ret["model_county_matches_geo_county"].fillna(False)
				& ret["model_hse_region_matches_spatial_hse_region"].fillna(False)
		)
	else:
		ret["hospital_model_match_accepted"] = (
				ret["hospital_model_match_score_accepted"]
				& ret["model_hse_region_matches_spatial_hse_region"].fillna(False)
		)

	ret["hospital_model_match_review_required"] = ~ret["hospital_model_match_accepted"]

	model_value_columns = [
		"hospital_model_number",
		"hospital_model_group",
		"is_model_4",
		"is_model_3",
		"is_model_2",
		"is_model_3_or_4",
		"is_model_2_to_4",
		"is_specialist_hospital",
		"specialist_type",
		"hospital_model_source",
	]

	ret.loc[~ret["hospital_model_match_accepted"], model_value_columns] = pd.NA

	boolean_columns = [
		"is_model_4",
		"is_model_3",
		"is_model_2",
		"is_model_3_or_4",
		"is_model_2_to_4",
		"is_specialist_hospital",
	]

	for column in boolean_columns:
		ret[column] = ret[column].fillna(False).astype(bool)

	match_audit = pd.DataFrame(ret.drop(columns="geometry", errors="ignore"))[
		[
			"hospital_id",
			"hospital_name",
			"county_english",
			"hse_region",
			"model_hospital_name",
			"model_hse_region",
			"model_county_raw",
			"model_county_clean",
			"hospital_model_number",
			"hospital_model_group",
			"specialist_type",
			"hospital_model_match_score",
			"hospital_model_match_score_accepted",
			"model_county_matches_geo_county",
			"model_hse_region_matches_spatial_hse_region",
			"hospital_model_match_accepted",
			"hospital_model_match_review_required",
		]
	].copy()

	return ret, match_audit


def summarise_public_acute_capability_by_county(public_acute_hospitals: gpd.GeoDataFrame) -> pd.DataFrame:
	#### Summarise public acute hospital model capability by source county.
	df = public_acute_hospitals.copy()

	for column in ["is_model_4",
	               "is_model_3",
	               "is_model_2",
	               "is_model_3_or_4",
	               "is_model_2_to_4",
	               "is_specialist_hospital", ]:
		df[column] = df[column].fillna(False).astype(bool)

	ret = (df.groupby(["model_county_raw", "model_county_clean"],
	                  as_index=False,
	                  dropna=False,
	                  observed=True, )
	       .agg(n_public_acute_hospitals=("public_acute_hospital_id", "nunique"),
	            n_model_4_hospitals=("is_model_4", "sum"),
	            n_model_3_hospitals=("is_model_3", "sum"),
	            n_model_2_hospitals=("is_model_2", "sum"),
	            n_model_3_or_4_hospitals=("is_model_3_or_4", "sum"),
	            n_model_2_to_4_hospitals=("is_model_2_to_4", "sum"),
	            n_specialist_hospitals=("is_specialist_hospital", "sum"),
	            n_matched_to_geohive_points=("public_acute_match_accepted", "sum"), )
	       .rename(columns={"model_county_raw": "county",
	                        "model_county_clean": "county_lookup_key", })
	       .sort_values("county")
	       .reset_index(drop=True))

	return ret


def join_public_acute_names_where(df: pd.DataFrame,
                                  index,
                                  flag_column: str,
                                  name_column: str = "model_hospital_name",
                                  ) -> str:
	#### Join public acute hospital names within a group where a Boolean flag is true.
	group = df.loc[index].copy()

	ret = "; ".join(sorted(group.loc[group[flag_column].fillna(False),
	name_column].dropna().astype(str).unique()))

	return ret


def summarise_public_acute_capability_by_hse_region(public_acute_hospitals: gpd.GeoDataFrame) -> pd.DataFrame:
	#### Summarise public acute hospital model capability by source HSE health region.
	df = public_acute_hospitals.copy()

	for column in ["is_model_4",
	               "is_model_3",
	               "is_model_2",
	               "is_model_3_or_4",
	               "is_model_2_to_4",
	               "is_specialist_hospital", ]:
		df[column] = df[column].fillna(False).astype(bool)

	ret = (df.groupby("model_hse_region",
	                  as_index=False,
	                  dropna=False,
	                  observed=True, )
	       .agg(n_public_acute_hospitals=("public_acute_hospital_id", "nunique"),
	            n_model_4_hospitals=("is_model_4", "sum"),
	            n_model_3_hospitals=("is_model_3", "sum"),
	            n_model_2_hospitals=("is_model_2", "sum"),
	            n_model_3_or_4_hospitals=("is_model_3_or_4", "sum"),
	            n_model_2_to_4_hospitals=("is_model_2_to_4", "sum"),
	            n_specialist_hospitals=("is_specialist_hospital", "sum"),
	            n_matched_to_geohive_points=("public_acute_match_accepted", "sum"), )
	       .rename(columns={"model_hse_region": "hse_region"}))

	name_rows = []

	for hse_region, group in df.groupby("model_hse_region", dropna=False, observed=True):
		name_rows.append({
			"hse_region": hse_region,
			"model_4_hospital_names": join_public_acute_names_where(df, group.index, "is_model_4"),
			"model_3_or_4_hospital_names": join_public_acute_names_where(df, group.index, "is_model_3_or_4"),
			"specialist_hospital_names": join_public_acute_names_where(df, group.index, "is_specialist_hospital"),
		})

	names = pd.DataFrame(name_rows)

	ret = (ret.merge(names,
	                 on="hse_region",
	                 how="left",
	                 validate="one_to_one")
	       .sort_values("hse_region")
	       .reset_index(drop=True))

	return ret


def write_hospital_geo_outputs(dirs: dict[str, Path],
							   logger,
							   small_areas: gpd.GeoDataFrame,
							   health_lookup: gpd.GeoDataFrame,
							   paths: ProjectPaths,
							   ) -> dict[str, Path]:
	#### Build and write GeoHive hospital geography and public acute hospital capability outputs.
	hospitals, hospital_models = load_hospital_inputs(paths)

	write_hospital_raw_audits(dirs=dirs,
							  logger=logger,
							  hospitals=hospitals,
							  hospital_models=hospital_models,)

	geohive_hospitals, county_match_audit = build_geohive_hospital_geographies(
		hospitals=hospitals,
		small_areas=small_areas,
		health_lookup=health_lookup,
		paths=paths,
	)

	geohive_hospital_table = pd.DataFrame(
		geohive_hospitals.drop(columns="geometry", errors="ignore")
	)

	hospital_output_path = save_dataset_with_manifest(
		df=geohive_hospital_table,
		output_path=dirs["processed"] / GEO_HOSPITALS_FILE,
		manifest_dir=dirs["manifests"],
		stage="geo_hospitals",
	)

	hospital_geometry_path = dirs["processed"] / GEO_HOSPITALS_GEOMETRY_FILE
	geohive_hospitals.to_file(hospital_geometry_path,
							  driver="GPKG",
							  layer="geo_hospitals_geometry",)

	county_match_audit_path = save_dataset_with_manifest(
		df=county_match_audit,
		output_path=dirs["processed"] / GEO_HOSPITAL_COUNTY_MATCH_AUDIT_FILE,
		manifest_dir=dirs["manifests"],
		stage="geo_hospital_county_match_audit",
	)

	public_acute_hospitals, public_acute_match_audit = build_public_acute_hospital_points(
		hospital_models=hospital_models,
		geohive_hospitals=geohive_hospitals,
		paths=paths,
	)

	public_acute_table = pd.DataFrame(
		public_acute_hospitals.drop(columns="geometry", errors="ignore")
	)

	public_acute_output_path = save_dataset_with_manifest(
		df=public_acute_table,
		output_path=dirs["processed"] / GEO_PUBLIC_ACUTE_HOSPITAL_POINTS_FILE,
		manifest_dir=dirs["manifests"],
		stage="geo_public_acute_hospital_points_2023",
	)

	public_acute_geometry = public_acute_hospitals.loc[
		public_acute_hospitals["public_acute_match_accepted"].fillna(False)
	].copy()

	public_acute_geometry_path = dirs["processed"] / GEO_PUBLIC_ACUTE_HOSPITAL_POINTS_GEOMETRY_FILE
	public_acute_geometry.to_file(public_acute_geometry_path,
								  driver="GPKG",
								  layer="geo_public_acute_hospital_points_2023",)

	public_acute_match_audit_path = save_dataset_with_manifest(
		df=public_acute_match_audit,
		output_path=dirs["processed"] / GEO_PUBLIC_ACUTE_HOSPITAL_MATCH_AUDIT_FILE,
		manifest_dir=dirs["manifests"],
		stage="geo_public_acute_hospital_match_audit",
	)

	county_summary = summarise_public_acute_capability_by_county(public_acute_hospitals)
	county_summary_path = save_dataset_with_manifest(
		df=county_summary,
		output_path=dirs["processed"] / GEO_HOSPITAL_CAPABILITY_SUMMARY_BY_COUNTY_FILE,
		manifest_dir=dirs["manifests"],
		stage="geo_hospital_capability_summary_by_county",
	)

	region_summary = summarise_public_acute_capability_by_hse_region(public_acute_hospitals)
	region_summary_path = save_dataset_with_manifest(
		df=region_summary,
		output_path=dirs["processed"] / GEO_HOSPITAL_CAPABILITY_SUMMARY_BY_HSE_REGION_FILE,
		manifest_dir=dirs["manifests"],
		stage="geo_hospital_capability_summary_by_hse_region",
	)

	write_column_audit(geohive_hospital_table,
					   dirs["manifests"] / "geo_hospitals_column_audit.csv",)

	write_column_audit(public_acute_table,
					   dirs["manifests"] / "geo_public_acute_hospital_points_2023_column_audit.csv",)

	log_block(logger,
			  "Hospital geography and public acute capability outputs saved",
			  {"geo_hospitals": str(hospital_output_path),
			   "geo_hospitals_geometry": str(hospital_geometry_path),
			   "geo_hospital_county_match_audit": str(county_match_audit_path),
			   "geo_public_acute_hospital_points_2023": str(public_acute_output_path),
			   "geo_public_acute_hospital_points_2023_geometry": str(public_acute_geometry_path),
			   "geo_public_acute_hospital_match_audit": str(public_acute_match_audit_path),
			   "geo_hospital_capability_summary_by_county": str(county_summary_path),
			   "geo_hospital_capability_summary_by_hse_region": str(region_summary_path),
			   "geohive_rows": int(geohive_hospital_table.shape[0]),
			   "public_acute_rows": int(public_acute_table.shape[0]),
			   "public_acute_matches_accepted": int(public_acute_table["public_acute_match_accepted"].sum()),
			   "public_acute_matches_review_required": int(public_acute_table["public_acute_match_review_required"].sum()),
			   "county_match_review_required": int(geohive_hospital_table["county_match_review_required"].sum()),},)

	return {
		"geo_hospitals": hospital_output_path,
		"geo_hospitals_geometry": hospital_geometry_path,
		"geo_hospital_county_match_audit": county_match_audit_path,
		"geo_public_acute_hospital_points_2023": public_acute_output_path,
		"geo_public_acute_hospital_points_2023_geometry": public_acute_geometry_path,
		"geo_public_acute_hospital_match_audit": public_acute_match_audit_path,
		"geo_hospital_capability_summary_by_county": county_summary_path,
		"geo_hospital_capability_summary_by_hse_region": region_summary_path,
	}