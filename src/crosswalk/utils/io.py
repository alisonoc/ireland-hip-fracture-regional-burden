#### io.py

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from crosswalk.preprocessing.reference import harmonize_hse_region_columns
from crosswalk.utils.manifest import (
	dataframe_manifest,
	write_column_audit,
	write_manifest,
)


def ensure_directory(path: Path) -> Path:
	#### Create one directory and return the path.
	path.mkdir(parents=True, exist_ok=True)

	return path


def ensure_directories(paths: list[Path] | dict[str, Path],
					   ) -> list[Path] | dict[str, Path]:
	#### Create multiple directories and return the original collection.
	if isinstance(paths, dict):
		for folder in paths.values():
			folder.mkdir(parents=True, exist_ok=True)

		return paths

	for folder in paths:
		folder.mkdir(parents=True, exist_ok=True)

	return paths


def read_required_csv(path: Path,
					  name: str,
					  date_columns: list[str] | None = None,
					  **read_csv_kwargs,
					  ) -> pd.DataFrame:
	#### Read a required CSV file and raise a clear error if it is missing.
	if not path.exists():
		raise FileNotFoundError(f"{name} file not found: {path}")

	ret = pd.read_csv(path,
					  parse_dates=date_columns,
					  **read_csv_kwargs,)

	return harmonize_hse_region_columns(ret)


def read_required_json(path: Path,
					   name: str,
					   ) -> dict | list:
	#### Read a required JSON file and raise a clear error if it is missing.
	if not path.exists():
		raise FileNotFoundError(f"{name} file not found: {path}")

	with open(path, "r", encoding="utf-8") as file:
		ret = json.load(file)

	return ret


def write_json(data: dict | list,
			   output_path: Path,
			   ) -> Path:
	#### Write dictionary or list data to a JSON file.
	output_path.parent.mkdir(parents=True, exist_ok=True)

	with open(output_path, "w", encoding="utf-8") as file:
		json.dump(data,
				  file,
				  indent=2,
				  ensure_ascii=False,)

	return output_path


def save_dataframe_csv(df: pd.DataFrame,
					   output_path: Path,
					   index: bool = False,
					   ) -> Path:
	#### Save a dataframe to CSV without writing audit metadata.
	output_path.parent.mkdir(parents=True, exist_ok=True)

	df = harmonize_hse_region_columns(df)

	df.to_csv(output_path,
			  index=index,)

	return output_path


def save_dataset_with_manifest(df: pd.DataFrame,
							   output_path: Path,
							   manifest_dir: Path,
							   stage: str,
							   audit_filename: str | None = None,
							   manifest_filename: str | None = None,
							   write_audit: bool = True,
							   write_json_manifest: bool = True,
							   index: bool = False,
							   ) -> Path:
	#### Save a dataframe and route audit metadata to a manifest directory.
	output_path.parent.mkdir(parents=True, exist_ok=True)
	manifest_dir.mkdir(parents=True, exist_ok=True)

	df = harmonize_hse_region_columns(df)

	df.to_csv(output_path,
			  index=index,)

	if audit_filename is None:
		audit_filename = f"{output_path.stem}_column_audit.csv"

	if manifest_filename is None:
		manifest_filename = f"{output_path.stem}.json"

	if write_audit:
		audit_path = manifest_dir / audit_filename
		write_column_audit(df, audit_path)

	if write_json_manifest:
		manifest_path = manifest_dir / manifest_filename
		manifest = dataframe_manifest(df=df,
									  output_path=output_path,
									  stage=stage,)
		write_manifest(manifest, manifest_path)

	return output_path


def save_diagnostic_table(df: pd.DataFrame,
						  output_path: Path,
						  manifest_dir: Path,
						  stage: str,
						  write_audit: bool = True,
						  write_json_manifest: bool = True,
						  ) -> Path:
	#### Save a diagnostic table and route metadata to a manifest directory.
	ret = save_dataset_with_manifest(df=df,
									 output_path=output_path,
									 manifest_dir=manifest_dir,
									 stage=stage,
									 write_audit=write_audit,
									 write_json_manifest=write_json_manifest,
									 index=False,)

	return ret


def save_audit_table(df: pd.DataFrame,
					 output_path: Path,
					 stage: str,
					 write_audit: bool = False,
					 write_json_manifest: bool = True,
					 ) -> Path:
	#### Save an audit table directly inside a manifest directory.
	output_path.parent.mkdir(parents=True, exist_ok=True)

	df.to_csv(output_path,
			  index=False,)

	if write_audit:
		audit_path = output_path.parent / f"{output_path.stem}_column_audit.csv"
		write_column_audit(df, audit_path)

	if write_json_manifest:
		manifest_path = output_path.parent / f"{output_path.stem}.json"
		manifest = dataframe_manifest(df=df,
									  output_path=output_path,
									  stage=stage,)
		write_manifest(manifest, manifest_path)

	return output_path


def save_reporting_table(df: pd.DataFrame,
						 output_path: Path,
						 index: bool = False,
						 ) -> Path:
	#### Save a selected reporting table without writing audit metadata.
	output_path.parent.mkdir(parents=True, exist_ok=True)

	df.to_csv(output_path,
			  index=index,)

	return output_path


def build_output_record(name: str,
						path: Path,
						output_type: str,
						stage: str,
						) -> dict:
	#### Build a standard output record for logging or summary tables.
	ret = {"name": name,
		   "path": str(path),
		   "output_type": output_type,
		   "stage": stage,}

	return ret


def summarise_output_paths(output_paths: dict[str, Path],
						   output_type: str,
						   stage: str,
						   ) -> pd.DataFrame:
	#### Convert a dictionary of output paths to a dataframe summary.
	ret = pd.DataFrame([build_output_record(name=name,
											path=path,
											output_type=output_type,
											stage=stage,)
						for name, path in output_paths.items()])

	return ret