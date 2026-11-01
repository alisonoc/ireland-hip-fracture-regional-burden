#### check_manuscript_tables.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from check_national_burden import check_national_burden
from check_regional_cases_and_bed_days import check_regional_bed_days, check_regional_cases
from table_checks import repository_root


BOXPLOT_FIGURE = "ihfd_asa_age_los_by_age_group_asa_grade_grouped_boxplot.png"


def confirm_boxplot_figure() -> None:
	#### Confirm the ASA-age boxplot is archived as a figure, without reading patient rows.
	path = repository_root() / "figures" / BOXPLOT_FIGURE
	if not path.is_file():
		raise FileNotFoundError(f"Missing boxplot figure: figures/{BOXPLOT_FIGURE}")
	if path.stat().st_size < 1000:
		raise ValueError(f"figures/{BOXPLOT_FIGURE} is too small to be the manuscript figure")


def main() -> None:
	#### Run the manuscript table checks.
	check_national_burden()
	print("National burden table matches the manuscript.")
	check_regional_cases()
	print("Regional case table matches the manuscript.")
	check_regional_bed_days()
	print("Regional bed-day table matches the manuscript.")
	confirm_boxplot_figure()
	print("ASA-age boxplot is present as a PNG.")


if __name__ == "__main__":
	main()
