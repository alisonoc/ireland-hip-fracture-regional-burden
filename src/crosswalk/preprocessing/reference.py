#### reference.py



from __future__ import annotations



import pandas as pd





METE_SEVERITY_MAPPING = {"YELLOW": 1,

						 "ORANGE": 2,

						 "RED": 3,}



METE_SEVERITY_ORDER = ["Yellow",

					   "Orange",

					   "Red",]



HSE_REGION_ORDER = ["HSE Dublin and North East",

					"HSE Dublin and South East",

					"HSE Dublin and Midlands",

					"HSE Mid West",

					"HSE South West",

					"HSE West and North West",]



HSE_REGION_NAME_ALIASES = {

	"HSE Midwest": "HSE Mid West",

	"HSE Mid-West": "HSE Mid West",

}



HSE_REGION_INLINE_REPLACEMENTS = (

	("Midwest", "Mid West"),

	("Mid-West", "Mid West"),

	("Mid west", "Mid West"),

)



HSE_REGION_COLUMNS = ("hse_region",)

HSE_REGION_GEOGRAPHY_COLUMNS = ("geography_name", "recommended_hse_region", "audit_value", "term_label",)

HSE_REGION_FORMULA_COLUMNS = ("term",)



SEX_KEEP = ["Male",

			"Female",]



AGE_65PLUS_GROUPS = ["65 - 69 years",

					 "70 - 74 years",

					 "75 - 79 years",

					 "80 - 84 years",

					 "85 years and over",]





def normalize_hse_region_name(region: str) -> str:

	#### Harmonise legacy HSE region labels to the canonical project names.

	text = str(region).strip()



	if text in HSE_REGION_NAME_ALIASES:

		return HSE_REGION_NAME_ALIASES[text]



	for old, new in HSE_REGION_INLINE_REPLACEMENTS:

		text = text.replace(old, new)



	return HSE_REGION_NAME_ALIASES.get(text, text)





def harmonize_hse_region_series(series: pd.Series,

								require_hse_prefix: bool = False,

								) -> pd.Series:

	#### Harmonise one HSE-region label column to canonical project names.

	def _harmonize_value(value):

		if pd.isna(value):

			return value



		text = str(value).strip()



		if require_hse_prefix and not text.startswith("HSE"):

			return value



		return normalize_hse_region_name(text)



	return series.map(_harmonize_value)





def harmonize_hse_region_columns(df: pd.DataFrame,

								 columns: list[str] | None = None,

								 ) -> pd.DataFrame:

	#### Harmonise HSE-region label columns in a dataframe copy.

	if df.empty:

		return df



	ret = df.copy()



	if columns is None:

		columns = [column for column in (*HSE_REGION_COLUMNS,

										 *HSE_REGION_GEOGRAPHY_COLUMNS,

										 *HSE_REGION_FORMULA_COLUMNS)

				   if column in ret.columns]



	for column in columns:

		require_hse_prefix = column in HSE_REGION_GEOGRAPHY_COLUMNS

		ret[column] = harmonize_hse_region_series(ret[column],

												  require_hse_prefix=require_hse_prefix,)



	return ret





def order_hse_regions(regions: pd.Index | list[str]) -> list[str]:

	#### Return HSE regions in the canonical project display order.

	order = {region: idx for idx, region in enumerate(HSE_REGION_ORDER)}

	unique = pd.Index(normalize_hse_region_name(region) for region in regions).unique().tolist()

	missing = sorted(set(unique) - set(HSE_REGION_ORDER))



	if missing:

		raise ValueError(f"HSE regions missing from HSE_REGION_ORDER: {missing}")



	return sorted(unique, key=lambda region: order[region])





def order_hse_region_rows(df: pd.DataFrame,

						  region_col: str = "hse_region",

						  ) -> pd.DataFrame:

	#### Sort rows by the canonical HSE region display order.

	ret = df.copy()

	ret[region_col] = ret[region_col].astype(str).map(normalize_hse_region_name)

	region_order = order_hse_regions(ret[region_col])

	ret[region_col] = pd.Categorical(ret[region_col].astype(str),

								   categories=region_order,

								   ordered=True,)



	return ret.sort_values(region_col).reset_index(drop=True)

