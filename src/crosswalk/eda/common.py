#### common.py

from __future__ import annotations

from pathlib import Path

import pandas as pd

from crosswalk.paths import ProjectPaths
from crosswalk.utils.io import (
	read_required_csv,
	save_diagnostic_table,
)
from crosswalk.utils.text import normalise_text_label
from crosswalk.utils.validation import require_columns

def read_processed_csv(paths: ProjectPaths,
					   filename: str,
					   stage: str = "preprocessing",
					   date_columns: list[str] | None = None,
					   **read_csv_kwargs,
					   ) -> pd.DataFrame:
	#### Read a processed WEATHERED CSV from a specified pipeline stage.
	dirs = paths.stage_dirs(stage)

	ret = read_required_csv(path=dirs["processed"] / filename,
							name=f"{stage} processed file {filename}",
							date_columns=date_columns,
							**read_csv_kwargs,)

	return ret


def read_eda_csv(paths: ProjectPaths,
				 filename: str,
				 date_columns: list[str] | None = None,
				 **read_csv_kwargs,
				 ) -> pd.DataFrame:
	#### Read a processed EDA CSV.
	ret = read_processed_csv(paths=paths,
							 filename=filename,
							 stage="eda",
							 date_columns=date_columns,
							 **read_csv_kwargs,)

	return ret


def read_modelling_csv(paths: ProjectPaths,
					   filename: str,
					   date_columns: list[str] | None = None,
					   **read_csv_kwargs,
					   ) -> pd.DataFrame:
	#### Read a processed modelling CSV.
	ret = read_processed_csv(paths=paths,
							 filename=filename,
							 stage="modelling",
							 date_columns=date_columns,
							 **read_csv_kwargs,)

	return ret

def write_eda_table(df: pd.DataFrame,
					output_path: Path,
					stage: str,
					manifest_dir: Path,
					write_audit: bool = True,
					write_json_manifest: bool = True,
					) -> Path:
	#### Write an EDA diagnostic table and route metadata to the EDA manifest directory.
	ret = save_diagnostic_table(df=df,
								output_path=output_path,
								manifest_dir=manifest_dir,
								stage=stage,
								write_audit=write_audit,
								write_json_manifest=write_json_manifest,)

	return ret


def write_eda_tables(outputs: dict[str, pd.DataFrame],
					 output_dir: Path,
					 manifest_dir: Path,
					 write_audit: bool = True,
					 write_json_manifest: bool = True,
					 ) -> dict[str, Path]:
	#### Write multiple EDA diagnostic tables using standard output names.
	ret = {}

	for key, df in outputs.items():
		ret[key] = write_eda_table(df=df,
								   output_path=output_dir / f"{key}.csv",
								   stage=key,
								   manifest_dir=manifest_dir,
								   write_audit=write_audit,
								   write_json_manifest=write_json_manifest,)

	return ret


def summarise_date_coverage(df: pd.DataFrame,
							date_col: str,
							dataset: str,
							) -> pd.DataFrame:
	#### Summarise date coverage for one processed dataset.
	require_columns(df=df,
					required={date_col},
					name=dataset,)

	dates = pd.to_datetime(df[date_col], errors="coerce")

	if dates.dropna().empty:
		ret = pd.DataFrame([{"dataset": dataset,
							 "date_column": date_col,
							 "min_date": pd.NA,
							 "max_date": pd.NA,
							 "n_rows": int(len(df)),
							 "n_valid_dates": 0,
							 "n_missing_dates": int(dates.isna().sum()),
							 "n_unique_dates": 0,},])

		return ret

	ret = pd.DataFrame([{"dataset": dataset,
						 "date_column": date_col,
						 "min_date": dates.min().date().isoformat(),
						 "max_date": dates.max().date().isoformat(),
						 "n_rows": int(len(df)),
						 "n_valid_dates": int(dates.notna().sum()),
						 "n_missing_dates": int(dates.isna().sum()),
						 "n_unique_dates": int(dates.dt.date.nunique()),},])

	return ret


def summarise_year_coverage(df: pd.DataFrame,
							year_col: str,
							dataset: str,
							) -> pd.DataFrame:
	#### Summarise year coverage for one processed dataset.
	require_columns(df=df,
					required={year_col},
					name=dataset,)

	years = pd.to_numeric(df[year_col], errors="coerce").dropna().astype(int)

	if years.empty:
		ret = pd.DataFrame([{"dataset": dataset,
							 "year_column": year_col,
							 "min_year": pd.NA,
							 "max_year": pd.NA,
							 "n_rows": int(len(df)),
							 "n_unique_years": 0,
							 "years": "",},])

		return ret

	ret = pd.DataFrame([{"dataset": dataset,
						 "year_column": year_col,
						 "min_year": int(years.min()),
						 "max_year": int(years.max()),
						 "n_rows": int(len(df)),
						 "n_unique_years": int(years.nunique()),
						 "years": ", ".join(str(year) for year in sorted(years.unique())),},])

	return ret


def summarise_date_overlap(left_df: pd.DataFrame,
						   right_df: pd.DataFrame,
						   left_date_col: str,
						   right_date_col: str,
						   left_name: str,
						   right_name: str,
						   check_name: str | None = None,
						   ) -> pd.DataFrame:
	#### Summarise shared and source-specific dates between two dated datasets.
	require_columns(df=left_df,
					required={left_date_col},
					name=left_name,)

	require_columns(df=right_df,
					required={right_date_col},
					name=right_name,)

	left_dates = set(pd.to_datetime(left_df[left_date_col], errors="coerce").dropna().dt.date)
	right_dates = set(pd.to_datetime(right_df[right_date_col], errors="coerce").dropna().dt.date)

	shared_dates = left_dates.intersection(right_dates)
	left_only_dates = left_dates.difference(right_dates)
	right_only_dates = right_dates.difference(left_dates)

	if check_name is None:
		check_name = f"{left_name}_{right_name}_date_overlap"

	ret = pd.DataFrame([{"check": check_name,
						 "left_dataset": left_name,
						 "right_dataset": right_name,
						 "left_date_column": left_date_col,
						 "right_date_column": right_date_col,
						 "left_min_date": min(left_dates).isoformat() if left_dates else pd.NA,
						 "left_max_date": max(left_dates).isoformat() if left_dates else pd.NA,
						 "right_min_date": min(right_dates).isoformat() if right_dates else pd.NA,
						 "right_max_date": max(right_dates).isoformat() if right_dates else pd.NA,
						 "n_left_dates": int(len(left_dates)),
						 "n_right_dates": int(len(right_dates)),
						 "n_shared_dates": int(len(shared_dates)),
						 "n_left_only_dates": int(len(left_only_dates)),
						 "n_right_only_dates": int(len(right_only_dates)),
						 "left_dates_with_right_share": len(shared_dates) / len(left_dates) if left_dates else pd.NA,
						 "right_dates_with_left_share": len(shared_dates) / len(right_dates) if right_dates else pd.NA,},])

	return ret


def summarise_year_overlap(left_df: pd.DataFrame,
						   right_df: pd.DataFrame,
						   left_year_col: str,
						   right_year_col: str,
						   left_name: str,
						   right_name: str,
						   check_name: str | None = None,
						   ) -> pd.DataFrame:
	#### Summarise shared and source-specific years between two annual datasets.
	require_columns(df=left_df,
					required={left_year_col},
					name=left_name,)

	require_columns(df=right_df,
					required={right_year_col},
					name=right_name,)

	left_years = set(pd.to_numeric(left_df[left_year_col], errors="coerce").dropna().astype(int))
	right_years = set(pd.to_numeric(right_df[right_year_col], errors="coerce").dropna().astype(int))

	shared_years = left_years.intersection(right_years)
	left_only_years = left_years.difference(right_years)
	right_only_years = right_years.difference(left_years)

	if check_name is None:
		check_name = f"{left_name}_{right_name}_year_overlap"

	ret = pd.DataFrame([{"check": check_name,
						 "left_dataset": left_name,
						 "right_dataset": right_name,
						 "left_year_column": left_year_col,
						 "right_year_column": right_year_col,
						 "left_years": ", ".join(str(year) for year in sorted(left_years)),
						 "right_years": ", ".join(str(year) for year in sorted(right_years)),
						 "shared_years": ", ".join(str(year) for year in sorted(shared_years)),
						 "left_only_years": ", ".join(str(year) for year in sorted(left_only_years)),
						 "right_only_years": ", ".join(str(year) for year in sorted(right_only_years)),
						 "n_left_years": int(len(left_years)),
						 "n_right_years": int(len(right_years)),
						 "n_shared_years": int(len(shared_years)),},])

	return ret


def value_presence_audit(left_values: pd.Series,
						 right_values: pd.Series,
						 left_name: str,
						 right_name: str,
						 value_col: str,
						 normalise: bool = True,
						 ) -> pd.DataFrame:
	#### Compare whether values are present across two sources.
	if normalise:
		left = set(normalise_text_label(left_values).dropna())
		right = set(normalise_text_label(right_values).dropna())
	else:
		left = set(left_values.dropna())
		right = set(right_values.dropna())

	all_values = sorted(left | right)

	ret = pd.DataFrame([{value_col: value,
						 f"in_{left_name}": value in left,
						 f"in_{right_name}": value in right,
						 "in_both": value in left and value in right,}
						for value in all_values])

	return ret


def add_check_status(df: pd.DataFrame,
					 pass_col: str,
					 status_col: str = "status",
					 ) -> pd.DataFrame:
	#### Add pass or warning status from a boolean check column.
	ret = df.copy()

	require_columns(df=ret,
					required={pass_col},
					name="status input",)

	ret[status_col] = ret[pass_col].map({True: "pass",
										 False: "warning",})

	return ret


def format_observed_months(months: pd.Series) -> str:
	#### Format observed month values as a comma-separated string.
	valid_months = pd.to_numeric(months, errors="coerce").dropna().astype(int)
	ret = ", ".join(str(value) for value in sorted(valid_months.unique()))

	return ret


def format_unique_values(values: pd.Series,
						 separator: str = "; ",
						 ) -> str:
	#### Format unique non-missing values as a sorted separated string.
	ret = separator.join(str(value) for value in sorted(values.dropna().astype(str).unique()))

	return ret


def summarise_year_month_coverage(df: pd.DataFrame,
								  date_col: str,
								  group_cols: list[str],
								  dataset: str,
								  expected_months: int = 12,
								  ) -> pd.DataFrame:
	#### Summarise year-month activity for a dated dataset.
	require_columns(df=df,
					required=set(group_cols + [date_col]),
					name=dataset,)

	work = df.copy()
	work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
	work["year"] = work[date_col].dt.year.astype("Int64")
	work["month"] = work[date_col].dt.month.astype("Int64")

	ret = (work.groupby(group_cols + ["year"],
						as_index=False,
						dropna=False,
						observed=True,)
		   .agg(first_observed_date=(date_col, "min"),
				last_observed_date=(date_col, "max"),
				n_observed_dates=(date_col, "nunique"),
				n_observed_months=("month", "nunique"),
				observed_months=("month", format_observed_months),
				n_rows=(date_col, "size"),)
		   .sort_values(group_cols + ["year"])
		   .reset_index(drop=True))

	ret["dataset"] = dataset
	ret["expected_months"] = expected_months
	ret["has_observed_rows_in_all_expected_months"] = ret["n_observed_months"] == expected_months
	ret["is_complete_calendar_year"] = ret["has_observed_rows_in_all_expected_months"]
	ret["coverage_interpretation_note"] = (
		"Year-month activity summarises observed dated rows. "
		"For event-only datasets, this cannot be used to infer complete data coverage."
	)

	return ret


def summarise_year_availability(df: pd.DataFrame,
								year_col: str,
								group_cols: list[str],
								dataset: str,
								) -> pd.DataFrame:
	#### Summarise year availability for annual datasets.
	require_columns(df=df,
					required=set(group_cols + [year_col]),
					name=dataset,)

	work = df.copy()
	work[year_col] = pd.to_numeric(work[year_col], errors="coerce").astype("Int64")

	ret = (work.groupby(group_cols + [year_col],
						as_index=False,
						dropna=False,
						observed=True,)
		   .agg(n_rows=(year_col, "size"),)
		   .sort_values(group_cols + [year_col])
		   .reset_index(drop=True)
		   .rename(columns={year_col: "year"}))

	ret["dataset"] = dataset
	ret["has_year"] = ret["year"].notna()

	return ret


def summarise_column_missingness(df: pd.DataFrame,
								 dataset: str,
								 ) -> pd.DataFrame:
	#### Summarise missingness across all columns in a dataframe.
	n_rows = int(df.shape[0])

	ret = pd.DataFrame({"dataset": dataset,
						"column": df.columns,
						"dtype": [str(dtype) for dtype in df.dtypes],
						"n_rows": n_rows,
						"n_missing": [int(df[column].isna().sum()) for column in df.columns],
						"n_non_missing": [int(df[column].notna().sum()) for column in df.columns],})

	ret["pct_missing"] = (ret["n_missing"] / ret["n_rows"] * 100).where(ret["n_rows"] > 0, pd.NA)

	ret = (ret.sort_values(["pct_missing", "column"],
						   ascending=[False, True],)
		   .reset_index(drop=True))

	return ret


def build_pivot(df: pd.DataFrame,
				index: str,
				columns: str,
				values: str,
				fill_value: int | float | None = 0,
				aggfunc: str = "sum",
				) -> pd.DataFrame:
	#### Build a simple pivot table for EDA summaries and heatmap inputs.
	require_columns(df=df,
					required={index,
							  columns,
							  values,},
					name="pivot input",)

	ret = (df.pivot_table(index=index,
						  columns=columns,
						  values=values,
						  aggfunc=aggfunc,
						  fill_value=fill_value,)
		   .sort_index())

	ret.columns.name = None
	ret = ret.reset_index()

	return ret