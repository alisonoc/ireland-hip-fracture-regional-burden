#### check_regional_cases_and_bed_days.py
import pandas as pd

from table_checks import read_derived_csv, report_mismatches, require_columns, round_half_up


CASE_FILE = "population_only_hip_fracture_projection_hse_region_summary.csv"
BED_DAY_FILE = "service_burden_hse_region_summary.csv"

# Manuscript table tab:projection_regional_summary.
# Counts are rounded to the nearest case. Absolute change is the rounded difference.
# Percent change is the unrounded difference as a percentage of the unrounded 2022 count.
MANUSCRIPT_REGIONAL_CASES = {
	"HSE Dublin and North East": {"cases_2022": 909, "cases_2042": 1980, "absolute_change": 1071, "percent_change": 117.9},
	"HSE Dublin and South East": {"cases_2022": 817, "cases_2042": 1797, "absolute_change": 980, "percent_change": 120.0},
	"HSE Dublin and Midlands": {"cases_2022": 689, "cases_2042": 1519, "absolute_change": 830, "percent_change": 120.6},
	"HSE Mid West": {"cases_2022": 342, "cases_2042": 713, "absolute_change": 371, "percent_change": 108.7},
	"HSE South West": {"cases_2022": 586, "cases_2042": 1245, "absolute_change": 659, "percent_change": 112.5},
	"HSE West and North West": {"cases_2022": 676, "cases_2042": 1367, "absolute_change": 692, "percent_change": 102.4},
}

# Manuscript table tab:service_burden_regional.
MANUSCRIPT_REGIONAL_BED_DAYS = {
	"HSE Dublin and North East": {"bed_days_2022": 17223, "bed_days_2042": 37956, "absolute_change": 20733, "percent_change": 120.4},
	"HSE Dublin and South East": {"bed_days_2022": 16592, "bed_days_2042": 37170, "absolute_change": 20577, "percent_change": 124.0},
	"HSE Dublin and Midlands": {"bed_days_2022": 15207, "bed_days_2042": 34084, "absolute_change": 18878, "percent_change": 124.1},
	"HSE Mid West": {"bed_days_2022": 4623, "bed_days_2042": 9803, "absolute_change": 5180, "percent_change": 112.0},
	"HSE South West": {"bed_days_2022": 9475, "bed_days_2042": 20543, "absolute_change": 11068, "percent_change": 116.8},
	"HSE West and North West": {"bed_days_2022": 13602, "bed_days_2042": 28104, "absolute_change": 14502, "percent_change": 106.6},
}


def manuscript_projection_scenario(frame: pd.DataFrame, table_name: str) -> pd.DataFrame:
	#### Keep the M1 overall-observed-rate rows used in the manuscript.
	require_columns(frame, {"projection_method", "rate_scenario", "hse_region"}, table_name)
	selected = frame.loc[
		frame["projection_method"].eq("M1") & frame["rate_scenario"].eq("overall_observed_rate")
	].copy()
	if selected.empty:
		raise ValueError(f"{table_name} has no M1 overall_observed_rate rows")
	return selected


def rounded_change(baseline: float, latest: float) -> dict[str, float]:
	#### Round a 2022-2042 change the way the manuscript table does.
	return {
		"baseline": round_half_up(baseline, 0),
		"latest": round_half_up(latest, 0),
		"absolute_change": round_half_up(latest - baseline, 0),
		"percent_change": round_half_up((latest - baseline) / baseline * 100, 1),
	}


def one_value_per_region(frame: pd.DataFrame, column: str, table_name: str) -> pd.Series:
	#### Return one numeric value per region, after confirming duplicate rows agree.
	if frame.groupby("hse_region")[column].nunique().max() != 1:
		raise ValueError(f"{table_name} has conflicting {column} values for one region")
	return frame.groupby("hse_region")[column].first()


def compare_region_table(actual: dict[str, dict[str, float]],
						expected: dict[str, dict[str, float]],
						table_name: str,) -> None:
	#### Compare rounded regional rows with the manuscript.
	mismatches: list[str] = []
	missing = sorted(set(expected).difference(actual))
	extra = sorted(set(actual).difference(expected))
	for region in missing:
		mismatches.append(f"{region}: missing from the derived table")
	for region in extra:
		mismatches.append(f"{region}: not in the manuscript table")
	for region, expected_row in expected.items():
		if region not in actual:
			continue
		for key, expected_value in expected_row.items():
			actual_value = actual[region][key]
			if actual_value != expected_value:
				mismatches.append(
					f"{region} {key}: csv {actual_value} != manuscript {expected_value}"
				)
	report_mismatches(mismatches, table_name)


def check_regional_cases() -> None:
	#### Check the 2022 and 2042 regional case counts against the manuscript.
	frame = manuscript_projection_scenario(read_derived_csv(CASE_FILE), CASE_FILE)
	require_columns(
		frame,
		{
			"hse_region",
			"baseline_projection_year",
			"latest_projection_year",
			"baseline_projected_hip_fracture_count",
			"latest_projected_hip_fracture_count",
		},
		CASE_FILE,
	)
	if not frame["baseline_projection_year"].eq(2022).all():
		raise ValueError(f"{CASE_FILE} baseline year is not 2022 for every M1 row")
	if not frame["latest_projection_year"].eq(2042).all():
		raise ValueError(f"{CASE_FILE} latest year is not 2042 for every M1 row")
	baseline = one_value_per_region(frame, "baseline_projected_hip_fracture_count", CASE_FILE)
	latest = one_value_per_region(frame, "latest_projected_hip_fracture_count", CASE_FILE)
	actual: dict[str, dict[str, float]] = {}
	for region in baseline.index:
		rounded = rounded_change(float(baseline[region]), float(latest[region]))
		actual[region] = {
			"cases_2022": rounded["baseline"],
			"cases_2042": rounded["latest"],
			"absolute_change": rounded["absolute_change"],
			"percent_change": rounded["percent_change"],
		}
	compare_region_table(actual, MANUSCRIPT_REGIONAL_CASES, "Regional case table")


def check_regional_bed_days() -> None:
	#### Check the 2022 and 2042 regional bed-day totals against the manuscript.
	frame = manuscript_projection_scenario(read_derived_csv(BED_DAY_FILE), BED_DAY_FILE)
	require_columns(
		frame,
		{
			"hse_region",
			"los_scenario",
			"baseline_projection_year_bed_day",
			"latest_projection_year_bed_day",
			"baseline_projected_bed_days",
			"latest_projected_bed_days",
		},
		BED_DAY_FILE,
	)
	frame = frame.loc[frame["los_scenario"].eq("region_age_mean_los_with_fallback")].copy()
	if frame.empty:
		raise ValueError(f"{BED_DAY_FILE} has no region_age_mean_los_with_fallback rows")
	if not frame["baseline_projection_year_bed_day"].eq(2022).all():
		raise ValueError(f"{BED_DAY_FILE} baseline year is not 2022")
	if not frame["latest_projection_year_bed_day"].eq(2042).all():
		raise ValueError(f"{BED_DAY_FILE} latest year is not 2042")
	baseline = one_value_per_region(frame, "baseline_projected_bed_days", BED_DAY_FILE)
	latest = one_value_per_region(frame, "latest_projected_bed_days", BED_DAY_FILE)
	actual: dict[str, dict[str, float]] = {}
	for region in baseline.index:
		rounded = rounded_change(float(baseline[region]), float(latest[region]))
		actual[region] = {
			"bed_days_2022": rounded["baseline"],
			"bed_days_2042": rounded["latest"],
			"absolute_change": rounded["absolute_change"],
			"percent_change": rounded["percent_change"],
		}
	compare_region_table(actual, MANUSCRIPT_REGIONAL_BED_DAYS, "Regional bed-day table")
