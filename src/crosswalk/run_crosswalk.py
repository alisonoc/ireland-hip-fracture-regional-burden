#### run_crosswalk.py
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from crosswalk.eda.cso_hhr_validation import run_cso_hhr_validation
from crosswalk.paths import get_project_paths
from crosswalk.preprocessing.preprocess_cso import run_cso_preprocessing
from crosswalk.preprocessing.preprocess_cso_projections import run_cso_projection_preprocessing
from crosswalk.preprocessing.preprocess_geo import run_geo_preprocessing


def main() -> None:
	#### Rebuild the county-to-health-region crosswalk from the local CSO and geography downloads.
	paths = get_project_paths()
	run_geo_preprocessing(paths)
	run_cso_preprocessing(paths)
	run_cso_projection_preprocessing(paths)
	run_cso_hhr_validation(paths)
	print("Crosswalk rebuild completed.")


if __name__ == "__main__":
	main()
