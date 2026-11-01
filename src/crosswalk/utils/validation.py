#### validation.py

from __future__ import annotations

import numpy as np
import pandas as pd


def require_columns(df: pd.DataFrame,
					required: set[str],
					name: str,
					) -> None:
	#### Check that a dataframe contains required columns.
	missing = required - set(df.columns)

	if missing:
		raise ValueError(f"{name} missing required columns: {missing}")


def validate_non_empty(df: pd.DataFrame,
					   name: str,
					   ) -> None:
	#### Check that a dataframe contains at least one row.
	if df.empty:
		raise ValueError(f"{name} is empty.")


def validate_unique_rows(df: pd.DataFrame,
						 keys: list[str],
						 name: str,
						 max_examples: int = 10,
						 ) -> None:
	#### Validate row uniqueness for a specified dataframe grain.
	require_columns(df=df,
					required=set(keys),
					name=name,)

	if df[keys].duplicated().any():
		duplicates = (df.loc[df[keys].duplicated(keep=False), keys]
					  .drop_duplicates()
					  .head(max_examples)
					  .to_dict(orient="records"))

		raise ValueError(f"{name} has duplicate rows for keys {keys}: {duplicates}")


def validate_no_missing(df: pd.DataFrame,
						columns: list[str],
						name: str,
						) -> None:
	#### Validate that selected columns contain no missing values.
	require_columns(df=df,
					required=set(columns),
					name=name,)

	missing_summary = {column: int(df[column].isna().sum())
					   for column in columns
					   if df[column].isna().any()}

	if missing_summary:
		raise ValueError(f"{name} contains missing values: {missing_summary}")


def validate_numeric_columns(df: pd.DataFrame,
							 columns: list[str],
							 name: str,
							 allow_missing: bool = False,
							 ) -> None:
	#### Validate that selected columns can be interpreted as numeric.
	require_columns(df=df,
					required=set(columns),
					name=name,)

	bad_columns = {}

	for column in columns:
		values = pd.to_numeric(df[column], errors="coerce")

		if not allow_missing and values.isna().any():
			bad_columns[column] = int(values.isna().sum())

		if allow_missing:
			new_missing = values.isna() & df[column].notna()

			if new_missing.any():
				bad_columns[column] = int(new_missing.sum())

	if bad_columns:
		raise ValueError(f"{name} contains non-numeric values: {bad_columns}")


def validate_non_negative(df: pd.DataFrame,
						  columns: list[str],
						  name: str,
						  allow_missing: bool = False,
						  ) -> None:
	#### Validate that selected numeric columns contain no negative values.
	require_columns(df=df,
					required=set(columns),
					name=name,)

	bad_columns = {}

	for column in columns:
		values = pd.to_numeric(df[column], errors="coerce")

		if not allow_missing and values.isna().any():
			bad_columns[column] = {"n_missing_or_non_numeric": int(values.isna().sum())}
			continue

		if allow_missing:
			values = values.dropna()

		if values.lt(0).any():
			bad_columns[column] = {"n_negative": int(values.lt(0).sum())}

	if bad_columns:
		raise ValueError(f"{name} contains invalid negative values: {bad_columns}")


def validate_positive(df: pd.DataFrame,
					  columns: list[str],
					  name: str,
					  allow_missing: bool = False,
					  ) -> None:
	#### Validate that selected numeric columns contain positive values only.
	require_columns(df=df,
					required=set(columns),
					name=name,)

	bad_columns = {}

	for column in columns:
		values = pd.to_numeric(df[column], errors="coerce")

		if not allow_missing and values.isna().any():
			bad_columns[column] = {"n_missing_or_non_numeric": int(values.isna().sum())}
			continue

		if allow_missing:
			values = values.dropna()

		if values.le(0).any():
			bad_columns[column] = {"n_non_positive": int(values.le(0).sum())}

	if bad_columns:
		raise ValueError(f"{name} contains non-positive values: {bad_columns}")


def validate_binary_column(df: pd.DataFrame,
						   column: str,
						   name: str,
						   allow_missing: bool = False,
						   ) -> None:
	#### Validate that one column contains only binary values.
	require_columns(df=df,
					required={column},
					name=name,)

	values = df[column]

	if allow_missing:
		values = values.dropna()
	elif values.isna().any():
		raise ValueError(f"{name} contains missing values in binary column {column}.")

	unique_values = set(values.dropna().unique())

	if not unique_values.issubset({0, 1, False, True}):
		raise ValueError(f"{name} contains non-binary values in {column}: {sorted(unique_values)}")


def validate_values_in_set(df: pd.DataFrame,
						   column: str,
						   allowed_values: set,
						   name: str,
						   allow_missing: bool = False,
						   ) -> None:
	#### Validate that one column contains only allowed values.
	require_columns(df=df,
					required={column},
					name=name,)

	values = df[column]

	if allow_missing:
		values = values.dropna()
	elif values.isna().any():
		raise ValueError(f"{name} contains missing values in {column}.")

	observed = set(values.dropna().unique())
	unexpected = observed - allowed_values

	if unexpected:
		raise ValueError(f"{name} contains unexpected values in {column}: {sorted(unexpected)}")


def validate_finite_numeric(df: pd.DataFrame,
							columns: list[str],
							name: str,
							allow_missing: bool = False,
							) -> None:
	#### Validate that selected numeric columns contain finite values.
	require_columns(df=df,
					required=set(columns),
					name=name,)

	bad_columns = {}

	for column in columns:
		values = pd.to_numeric(df[column], errors="coerce")

		if allow_missing:
			values = values.dropna()
		elif values.isna().any():
			bad_columns[column] = {"n_missing_or_non_numeric": int(values.isna().sum())}
			continue

		if not np.isfinite(values).all():
			bad_columns[column] = {"n_non_finite": int((~np.isfinite(values)).sum())}

	if bad_columns:
		raise ValueError(f"{name} contains non-finite numeric values: {bad_columns}")


def coerce_numeric_columns(df: pd.DataFrame,
						   columns: list[str],
						   name: str,
						   allow_missing: bool = False,
						   ) -> pd.DataFrame:
	#### Coerce selected columns to numeric and optionally reject missing values.
	require_columns(df=df,
					required=set(columns),
					name=name,)

	ret = df.copy()

	for column in columns:
		ret[column] = pd.to_numeric(ret[column], errors="coerce")

	validate_numeric_columns(df=ret,
							 columns=columns,
							 name=name,
							 allow_missing=allow_missing,)

	return ret

def validate_numeric_range(df: pd.DataFrame,
						   columns: list[str],
						   name: str,
						   lower: float | None = None,
						   upper: float | None = None,
						   allow_missing: bool = False,
						   inclusive: bool = True,
						   ) -> None:
	#### Validate that selected numeric columns fall within a specified range.
	require_columns(df=df,
					required=set(columns),
					name=name,)

	bad_columns = {}

	for column in columns:
		values = pd.to_numeric(df[column], errors="coerce")

		if allow_missing:
			values = values.dropna()
		elif values.isna().any():
			bad_columns[column] = {"n_missing_or_non_numeric": int(values.isna().sum())}
			continue

		if lower is not None:
			if inclusive:
				bad_lower = values.lt(lower)
			else:
				bad_lower = values.le(lower)

			if bad_lower.any():
				bad_columns.setdefault(column, {})
				bad_columns[column]["n_below_lower_bound"] = int(bad_lower.sum())

		if upper is not None:
			if inclusive:
				bad_upper = values.gt(upper)
			else:
				bad_upper = values.ge(upper)

			if bad_upper.any():
				bad_columns.setdefault(column, {})
				bad_columns[column]["n_above_upper_bound"] = int(bad_upper.sum())

	if bad_columns:
		raise ValueError(f"{name} contains values outside the permitted range: {bad_columns}")

def validate_grouped_sum(df: pd.DataFrame,
						 group_cols: list[str],
						 value_col: str,
						 name: str,
						 expected_sum: float = 1.0,
						 tolerance: float = 1e-6,
						 allow_missing: bool = False,
						 max_examples: int = 10,
						 ) -> None:
	#### Validate that a numeric column sums to an expected value within groups.
	require_columns(df=df,
					required=set(group_cols + [value_col]),
					name=name,)

	values = df.copy()
	values[value_col] = pd.to_numeric(values[value_col], errors="coerce")

	if not allow_missing and values[value_col].isna().any():
		raise ValueError(f"{name} contains missing or non-numeric values in {value_col}.")

	if allow_missing:
		values = values.dropna(subset=[value_col]).copy()

	grouped = (values.groupby(group_cols,
							  as_index=False,
							  dropna=False,
							  observed=True,)
			   .agg(observed_sum=(value_col, "sum"),))

	grouped["absolute_difference"] = (grouped["observed_sum"] - expected_sum).abs()

	bad = grouped.loc[grouped["absolute_difference"].gt(tolerance)].copy()

	if not bad.empty:
		examples = bad.head(max_examples).to_dict(orient="records")
		raise ValueError(f"{name} has grouped sums outside tolerance for {value_col}: {examples}")

def validate_required_dict_keys(config: dict,
								required_keys: set[str],
								name: str,
								) -> None:
	#### Validate that a dictionary contains required keys.
	missing = required_keys - set(config.keys())

	if missing:
		raise ValueError(f"{name} missing required keys: {missing}")

def validate_required_values_present(df: pd.DataFrame,
									 column: str,
									 required_values: set,
									 name: str,
									 allow_missing: bool = False,
									 ) -> None:
	#### Validate that required values are present in one dataframe column.
	require_columns(df=df,
					required={column},
					name=name,)

	values = df[column]

	if not allow_missing and values.isna().any():
		raise ValueError(f"{name} contains missing values in {column}.")

	observed_values = set(values.dropna().unique())
	missing_values = required_values - observed_values

	if missing_values:
		raise ValueError(f"{name} is missing required values in {column}: {sorted(missing_values)}")