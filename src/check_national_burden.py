#### check_national_burden.py
import pandas as pd

from table_checks import read_derived_csv, report_mismatches, require_columns, round_half_up


RELEASE_FILE = "ihfd_service_pathway_summary_release.csv"

ALLOWED_LEVELS = {
	"national_year",
	"national_broad_age_year",
	"national_age_year",
	"hse_region_year",
}

MINIMUM_CASES = {
	"national_year": 2024,
	"national_broad_age_year": 233,
	"national_age_year": 83,
	"hse_region_year": 32,
}

# Manuscript table tab:ihfd_national_burden. One decimal place on means, percentiles, and shares.
MANUSCRIPT_NATIONAL_BURDEN = [
	{"year": 2013, "cases": 2024, "bed_days": 42636, "mean_los": 21.1, "median_los": 14.0, "p75": 22.0, "p95": 61.0, "share_21": 27.9, "share_30": 14.5},
	{"year": 2014, "cases": 2737, "bed_days": 54794, "mean_los": 20.0, "median_los": 13.0, "p75": 21.0, "p95": 60.0, "share_21": 25.5, "share_30": 14.3},
	{"year": 2015, "cases": 3003, "bed_days": 60988, "mean_los": 20.3, "median_los": 13.0, "p75": 22.0, "p95": 61.9, "share_21": 28.1, "share_30": 16.8},
	{"year": 2016, "cases": 3175, "bed_days": 64483, "mean_los": 20.3, "median_los": 12.0, "p75": 21.0, "p95": 63.0, "share_21": 26.2, "share_30": 16.6},
	{"year": 2017, "cases": 3504, "bed_days": 69106, "mean_los": 19.7, "median_los": 13.0, "p75": 22.0, "p95": 61.0, "share_21": 27.2, "share_30": 16.4},
	{"year": 2018, "cases": 3755, "bed_days": 70527, "mean_los": 18.8, "median_los": 12.0, "p75": 21.0, "p95": 57.0, "share_21": 25.5, "share_30": 14.2},
	{"year": 2019, "cases": 3692, "bed_days": 71557, "mean_los": 19.4, "median_los": 12.0, "p75": 21.0, "p95": 63.0, "share_21": 25.3, "share_30": 15.5},
	{"year": 2020, "cases": 3634, "bed_days": 60141, "mean_los": 16.5, "median_los": 11.0, "p75": 18.8, "p95": 48.0, "share_21": 21.0, "share_30": 12.4},
	{"year": 2021, "cases": 3787, "bed_days": 65855, "mean_los": 17.4, "median_los": 12.0, "p75": 20.0, "p95": 52.0, "share_21": 24.0, "share_30": 12.8},
	{"year": 2022, "cases": 4002, "bed_days": 78339, "mean_los": 19.6, "median_los": 13.0, "p75": 22.0, "p95": 57.0, "share_21": 27.9, "share_30": 15.8},
	{"year": 2023, "cases": 3933, "bed_days": 72393, "mean_los": 18.4, "median_los": 12.0, "p75": 21.0, "p95": 56.0, "share_21": 25.4, "share_30": 14.5},
]


def guard_release_table(frame: pd.DataFrame) -> None:
	#### Reject a service-pathway file that still contains small cells.
	require_columns(frame, {"summary_level", "year", "n_cases"}, RELEASE_FILE)
	found = set(frame["summary_level"].dropna().unique())
	unexpected = sorted(found.difference(ALLOWED_LEVELS))
	if unexpected:
		raise ValueError(
			f"{RELEASE_FILE} contains summary levels that are held back: {unexpected}"
		)
	missing = sorted(set(MINIMUM_CASES).difference(found))
	if missing:
		raise ValueError(f"{RELEASE_FILE} is missing summary levels: {missing}")
	for level, minimum in MINIMUM_CASES.items():
		smallest = int(frame.loc[frame["summary_level"].eq(level), "n_cases"].min())
		if smallest < minimum:
			raise ValueError(
				f"{RELEASE_FILE} {level} has a cell of {smallest} cases; the release minimum is {minimum}"
			)


def rounded_national_row(row: pd.Series) -> dict[str, float]:
	#### Round one national year to the precision printed in the manuscript.
	return {
		"year": int(row["year"]),
		"cases": int(row["n_cases"]),
		"bed_days": round_half_up(row["total_bed_days"], 0),
		"mean_los": round_half_up(row["mean_los"], 1),
		"median_los": round_half_up(row["median_los"], 1),
		"p75": round_half_up(row["los_p75"], 1),
		"p95": round_half_up(row["los_p95"], 1),
		"share_21": round_half_up(row["share_long_stay_21plus"], 1),
		"share_30": round_half_up(row["share_long_stay_30plus"], 1),
	}


def check_national_burden() -> None:
	#### Check the 2013-2023 national burden table against the manuscript.
	frame = read_derived_csv(RELEASE_FILE)
	guard_release_table(frame)
	require_columns(
		frame,
		{
			"summary_level",
			"year",
			"n_cases",
			"total_bed_days",
			"mean_los",
			"median_los",
			"los_p75",
			"los_p95",
			"share_long_stay_21plus",
			"share_long_stay_30plus",
		},
		RELEASE_FILE,
	)
	national = frame.loc[
		frame["summary_level"].eq("national_year") & frame["year"].between(2013, 2023)
	].copy()
	if len(national) != len(MANUSCRIPT_NATIONAL_BURDEN):
		raise ValueError(
			f"Expected {len(MANUSCRIPT_NATIONAL_BURDEN)} national years, found {len(national)}"
		)
	by_year = {
		int(row["year"]): rounded_national_row(row)
		for _, row in national.iterrows()
	}
	mismatches: list[str] = []
	for expected in MANUSCRIPT_NATIONAL_BURDEN:
		year = expected["year"]
		if year not in by_year:
			mismatches.append(f"{year}: missing from {RELEASE_FILE}")
			continue
		for key, expected_value in expected.items():
			actual_value = by_year[year][key]
			if actual_value != expected_value:
				mismatches.append(
					f"{year} {key}: csv {actual_value} != manuscript {expected_value}"
				)
	report_mismatches(mismatches, "National burden table")
