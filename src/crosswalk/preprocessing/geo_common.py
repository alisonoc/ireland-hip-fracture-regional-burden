#### geo_common.py

from __future__ import annotations

import pandas as pd

from crosswalk.paths import ProjectPaths
from crosswalk.utils.io import read_required_json
from crosswalk.utils.text import normalise_lookup_label
from crosswalk.utils.validation import require_columns


GEO_RAW_INPUTS_SCHEMA_FILE = "geo_raw_inputs_schema.json"
GEO_REFERENCE_MAPPINGS_FILE = "geo_reference_mappings.json"


def load_geo_input_schema(paths: ProjectPaths,
						  schema_name: str,
						  ) -> dict:
	#### Load one named schema from the combined geo raw input schema file.
	config = read_required_json(path=paths.schemas / GEO_RAW_INPUTS_SCHEMA_FILE,
								name="Geo raw input schemas",)

	if "schemas" not in config:
		raise KeyError(f"Geo raw input schema file is missing 'schemas': {paths.schemas / GEO_RAW_INPUTS_SCHEMA_FILE}")

	if schema_name not in config["schemas"]:
		available = sorted(config["schemas"].keys())
		raise KeyError(f"Unknown geo schema '{schema_name}'. Available schemas: {available}")

	ret = config["schemas"][schema_name]

	return ret


def load_geo_reference_mappings(paths: ProjectPaths) -> dict:
	#### Load controlled geography reference mappings.
	ret = read_required_json(path=paths.schemas / GEO_REFERENCE_MAPPINGS_FILE,
							 name="Geo reference mappings",)

	return ret


def load_county_gaeilge_to_english_mapping(paths: ProjectPaths) -> dict[str, str]:
	#### Load controlled Gaeilge-to-English county mapping.
	config = load_geo_reference_mappings(paths)

	if "county_mapping" not in config:
		raise KeyError(f"Geo reference mappings file is missing 'county_mapping': {paths.schemas / GEO_REFERENCE_MAPPINGS_FILE}")

	ret = config["county_mapping"]

	return ret


def load_hospital_model_group_mapping(paths: ProjectPaths) -> dict[str, str]:
	#### Load controlled hospital model group labels.
	config = load_geo_reference_mappings(paths)

	if "hospital_model_group_mapping" not in config:
		raise KeyError(f"Geo reference mappings file is missing 'hospital_model_group_mapping': {paths.schemas / GEO_REFERENCE_MAPPINGS_FILE}")

	ret = config["hospital_model_group_mapping"]

	return ret


def load_hse_region_name_aliases(paths: ProjectPaths) -> dict[str, str]:
	#### Load controlled HSE region name aliases.
	config = load_geo_reference_mappings(paths)

	ret = config.get("hse_region_name_aliases", {})

	return ret


def normalise_county_name(series: pd.Series) -> pd.Series:
	#### Standardise county names for geography preprocessing joins.
	ret = (series.astype("string")
		   .str.strip()
		   .str.replace("’", "'", regex=False)
		   .str.replace("&", "and", regex=False)
		   .str.replace(r"\s+", " ", regex=True))

	return ret


def normalise_region_name(series: pd.Series) -> pd.Series:
	#### Standardise HSE health-region names for geography preprocessing joins.
	ret = (series.astype("string")
		   .str.strip()
		   .str.replace(r"^.*\|\s*", "", regex=True)
		   .str.replace(r"\s*HR$", "", regex=True)
		   .str.replace("Midwest", "Mid West", regex=False)
		   .str.replace("Mid-West", "Mid West", regex=False)
		   .str.replace("Mid west", "Mid West", regex=False)
		   .str.replace("&", "and", regex=False)
		   .str.replace(r"\s+", " ", regex=True))

	return ret


def normalise_region_name_with_reference(series: pd.Series,
										 paths: ProjectPaths,
										 ) -> pd.Series:
	#### Standardise HSE health-region names using controlled aliases before generic normalisation.
	aliases = load_hse_region_name_aliases(paths)

	ret = series.astype("string").str.strip()
	ret = ret.replace(aliases)
	ret = normalise_region_name(ret)

	return ret


def build_county_match_lookup(paths: ProjectPaths) -> pd.DataFrame:
	#### Build canonical county lookup used for hospital and geography label matching.
	county_mapping = load_county_gaeilge_to_english_mapping(paths)

	ret = pd.DataFrame(
		[
			{
				"county_gaeilge": county_gaeilge,
				"county_english": county_english,
			}
			for county_gaeilge, county_english in county_mapping.items()
		]
	)

	ret["county_lookup_key"] = normalise_lookup_label(ret["county_english"])

	return ret


def validate_county_match_lookup(county_lookup: pd.DataFrame) -> pd.DataFrame:
	#### Validate canonical county lookup used for fuzzy and exact county matching.
	required = {
		"county_gaeilge",
		"county_english",
		"county_lookup_key",
	}

	require_columns(df=county_lookup,
					required=required,
					name="county_match_lookup",)

	ret = county_lookup.copy()

	for column in required:
		ret[column] = ret[column].astype("string").str.strip()

	if ret["county_lookup_key"].duplicated().any():
		duplicates = (ret.loc[ret["county_lookup_key"].duplicated(keep=False),
							   ["county_gaeilge", "county_english", "county_lookup_key"]]
					  .sort_values("county_lookup_key")
					  .to_dict(orient="records"))

		raise ValueError(f"County lookup contains duplicate county_lookup_key values: {duplicates}")

	return ret