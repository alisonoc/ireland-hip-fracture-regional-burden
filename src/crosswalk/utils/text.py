#### text.py

import unicodedata
import pandas as pd
from difflib import SequenceMatcher

def normalise_text_label(series: pd.Series) -> pd.Series:
	#### Normalise text labels for conservative linkage checks.
	ret = (series.astype("string")
		   .str.strip()
		   .str.replace(r"\s+", " ", regex=True)
		   )

	return ret


def normalise_lookup_label(series: pd.Series) -> pd.Series:
	#### Normalise labels for lookup joins by lowercasing and removing accents.
	ret = (series.astype("string")
		   .str.strip()
		   .str.lower()
		   )

	ret = ret.apply(
		lambda value: unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode("utf-8")
		if pd.notna(value) else value
	)

	ret = (ret.str.replace("co. ", "", regex=False)
		   .str.replace("county ", "", regex=False)
		   .str.replace(r"\s+", " ", regex=True)
		   .str.strip()
		   )

	return ret

def extract_last_filled_text_field(df: pd.DataFrame,
								   columns: list[str],
								   ) -> pd.Series:
	#### Extract the last non-empty value across ordered text columns.
	work = df[columns].copy()

	for column in columns:
		work[column] = work[column].astype("string").str.strip()
		work[column] = work[column].replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})

	ret = work.apply(lambda row: row.dropna().iloc[-1] if not row.dropna().empty else pd.NA,
					 axis=1,)

	return ret.astype("string")


def normalise_county_candidate_label(series: pd.Series) -> pd.Series:
	#### Normalise county-like labels found in address fields.
	ret = normalise_lookup_label(series)

	ret = (ret.str.replace(r"\bdublin\s+\d+\b", "dublin", regex=True)
		   .str.replace(r"\bco\s+", "", regex=True)
		   .str.replace(r"\bcity\b", "", regex=True)
		   .str.replace(r"\d+", "", regex=True)
		   .str.replace(r"[^a-z\s]", " ", regex=True)
		   .str.replace(r"\s+", " ", regex=True)
		   .str.strip())

	return ret


def fuzzy_match_to_candidates(value: str,
							  candidates: list[str],
							  ) -> tuple[str | None, float]:
	#### Fuzzy-match one value to a controlled candidate list using a simple similarity score.
	if pd.isna(value) or str(value).strip() == "":
		return None, 0.0

	value_norm = str(value).strip().lower()

	best_label = None
	best_score = 0.0

	for candidate in candidates:
		candidate_norm = str(candidate).strip().lower()
		score = SequenceMatcher(None, value_norm, candidate_norm).ratio() * 100

		if score > best_score:
			best_label = candidate
			best_score = score

	return best_label, float(best_score)

