#### cso_age_schema.py
from crosswalk.utils.validation import validate_required_dict_keys


CSO_AGE_GROUP_SCHEMA_REQUIRED_KEYS = {
	"all_ages_label",
	"age_group_order",
	"analysis_groups",
}

CSO_ANALYSIS_GROUP_REQUIRED_KEYS = {
	"working_age_25_64",
	"age_80plus",
	"age_65plus",
	"age_60plus",
	"age_50plus",
}


def validate_cso_age_group_schema(age_group_schema: dict) -> None:
	#### Validate the CSO age-group schema used by the population crosswalk.
	validate_required_dict_keys(config=age_group_schema,
								required_keys=CSO_AGE_GROUP_SCHEMA_REQUIRED_KEYS,
								name="cso_age_groups_schema",)

	validate_required_dict_keys(config=age_group_schema["analysis_groups"],
								required_keys=CSO_ANALYSIS_GROUP_REQUIRED_KEYS,
								name="cso_age_groups_schema analysis_groups",)


def get_all_ages_label(age_group_schema: dict) -> str:
	#### Return the all-ages label from the CSO age-group schema.
	validate_cso_age_group_schema(age_group_schema)

	return age_group_schema["all_ages_label"]
