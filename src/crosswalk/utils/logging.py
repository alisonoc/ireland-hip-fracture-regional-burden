#### logging.py

import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo


def json_default(obj):
	#### Convert common scientific Python objects to JSON-safe values
	try:
		import numpy as np
		import pandas as pd

		if isinstance(obj, np.generic):
			return obj.item()

		if isinstance(obj, np.ndarray):
			return obj.tolist()

		if isinstance(obj, pd.Series):
			return obj.to_dict()

		if isinstance(obj, pd.DataFrame):
			return {
				"shape": list(obj.shape),
				"columns": list(map(str, obj.columns)),
			}

	except Exception:
		pass

	return str(obj)


def get_logger(
	name: str,
	log_dir: Path,
	filename: str | None = None,
	tz: str = "Europe/Dublin",
	to_console: bool = True,
	level: int = logging.INFO,
	mode: str = "a",
) -> logging.Logger:
	#### Create a file and console logger for WEATHERED scripts
	log_dir = Path(log_dir)
	log_dir.mkdir(parents=True, exist_ok=True)

	if filename is None:
		today = datetime.now(ZoneInfo(tz)).strftime("%Y%m%d")
		filename = f"{name}_{today}.log"

	if Path(filename).name != filename:
		raise ValueError("filename must not contain path separators")

	log_path = log_dir / filename

	logger = logging.getLogger(name)
	logger.setLevel(level)
	logger.propagate = False

	formatter = logging.Formatter(
		"%(asctime)s | %(levelname)s | %(name)s | %(message)s"
	)

	if to_console and not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
		stream_handler = logging.StreamHandler(sys.stdout)
		stream_handler.setFormatter(formatter)
		logger.addHandler(stream_handler)

	if not any(
		isinstance(h, logging.FileHandler)
		and getattr(h, "baseFilename", None) == str(log_path)
		for h in logger.handlers
	):
		file_handler = logging.FileHandler(log_path, mode=mode, encoding="utf-8")
		file_handler.setFormatter(formatter)
		logger.addHandler(file_handler)

	return logger


def log_block(logger: logging.Logger, title: str, obj) -> None:
	#### Log a JSON-formatted summary block
	body = json.dumps(
		obj,
		indent=2,
		ensure_ascii=False,
		default=json_default,
	)
	logger.info("\n==== %s ====\n%s", title, body)


def log_df(logger: logging.Logger, title: str, df, max_rows: int = 10) -> None:
	#### Log a compact dataframe preview
	if df is None:
		logger.info("\n==== %s: None ====", title)
		return

	if hasattr(df, "empty") and df.empty:
		logger.info("\n==== %s: empty ====", title)
		return

	if hasattr(df, "head"):
		logger.info(
			"\n==== %s | shape=%s ====\n%s",
			title,
			getattr(df, "shape", None),
			df.head(max_rows).to_string(index=False),
		)
	else:
		logger.info("\n==== %s ====\n%s", title, str(df))


def close_logger(logger: logging.Logger) -> None:
	#### Close and detach all handlers from a logger
	for handler in list(logger.handlers):
		logger.removeHandler(handler)
		try:
			handler.close()
		except Exception:
			pass