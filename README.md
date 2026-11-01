# Population ageing and future regional hip fracture service burden in the Republic of Ireland

Aggregate tables, public bed-stock extracts, figure files, and checking code for the manuscript submitted to *The Lancet Regional Health – Europe*.

**Population ageing and future regional hip fracture service burden in the Republic of Ireland (Ireland): implications for older-person care planning**

Alison N. O'Connor, Paul Harford, Louise Brent, Pamela Hickey, Damien Ryan, Fergal Cummins, Carrie Garavan, Gauri Vaidya, Meghana Kshirsagar

This is a spin-out from the WEATHERED project. Weather-warning analyses are outside this repository.

GitHub is the development copy. Cite the Zenodo archive of tag `v1.0.0-lancet-rh-europe`.

**Zenodo DOI:** [10.5281/zenodo.23859827](https://doi.org/10.5281/zenodo.23859827)

## Funding

Taighde Éireann – Research Ireland, National Challenge Fund, grant 22/NCF/DR/11289.

## What a reader can rebuild

Two separate checks live in this repository.

`python src/check_manuscript_tables.py` reads `data/derived/` and checks the manuscript counts for national burden and for regional cases and bed days in 2022 and 2042.

`python src/crosswalk/run_crosswalk.py` rebuilds the county-to-health-region crosswalk. It reads CSO population files and geography files that you place in `dataset/raw/`, using the same filenames as the original downloads. It writes the small-area weights, the health-region population master, the PEC26 projection tables, and the HHR comparison. It does not read Irish Hip Fracture Database records, and it does not rebuild the hip-fracture case projections. Those projections apply IHFD rates to the crosswalk, and the rate file stays in the private project.

The ASA–age boxplot is included as a figure file. The values behind that plot are patient-level length-of-stay records, which are not in this repository. The HHR validation panel in `figures/` is the archived figure. The crosswalk run writes the comparison tables and does not redraw that panel.

### Downloads the crosswalk run expects

Place these in `dataset/raw/`:

- `PEA04.20260504T110503.csv`
- `PEA08.20260504T110532.csv`
- `PEC26*.csv` and `PEC28*.csv`
- `HHR01*.csv` and `HHR02*.csv`
- `Small_Area_National_Statistical_Boundaries_2022_Ungeneralised_view_-7354763930310470674.gpkg`
- `Health_Geographies_(2025)_-_CSO_lookup.geojson`
- `CensusHub2022_T1_2_SA*.geojson`
- `Hospitals_-_HSE_Ireland.geojson`

Place `hse_hospital_model_numbers_2023.csv` in `dataset/published/esri_2025/`.

## Layout

```text
data/public/       Parliamentary bed-stock extracts
data/derived/      Aggregate tables used in the manuscript
data/RESTRICTED.md How to request patient-level IHFD access
dataset/raw/       CSO and geography downloads supplied by the reader
config/schemas/    Age-group and geography schemas
figures/           Manuscript figures
src/               Manuscript checks and the crosswalk package
```

## data/public

| File | Contents |
|---|---|
| `PQ_36610_23.csv` | Inpatient bed stock, April 2023, as released in parliamentary question 36610/23 |
| `PQ_12141_26.csv` | Inpatient bed stock, December 2025, as released in parliamentary question 12141/26 |

Central Statistics Office tables PEA03, PEA04, PEA08, PEC26, HHR01, and HHR02 are available from the publishers at [data.cso.ie](https://data.cso.ie/). Derived population and crosswalk tables are in `data/derived/`.

## data/derived

Population and geography:

- `cso_population_master_age_sex_region_county_year.csv`
- `cso_latest_demographic_pressure_by_region.csv`
- `cso_hhr_implied_hse_age_sex_2022_counts.csv`
- `cso_hhr_weathered_hse_age_sex_2022_validation.csv`
- `cso_hhr_validation_summary.csv`
- `geo_county_to_region_baseline.csv`

IHFD aggregates large enough to release:

- `ihfd_service_pathway_summary_release.csv`
- `ihfd_asa_age_los_logistic_regression.csv` (smallest cell: ASA V, 45 cases, national, 2013–2023 pooled)
- `ihfd_asa_age_los_pooled_log_linear_regression.csv`
- `ihfd_asa_age_los_pooled_log_linear_predicted_means.csv`
- `ihfd_asa_age_los_pooled_log_linear_marginal_effects.csv`
- `population_projection_2022_backtest.csv` (observed counts are national age groups; smallest is 192)
- `population_projection_regional_allocation_backtest.csv`
- `population_only_hip_fracture_projection_national_age_group_summary.csv`
- `population_only_hip_fracture_projection_hse_region_summary.csv`
- `population_projection_hse_region_age_group_2022_2042_manuscript_table.csv` (projected counts)
- `service_burden_hse_region_summary.csv`
- `episode_cost_price_path_hse_region.csv`

`ihfd_service_pathway_summary_release.csv` keeps four summary levels only:

| summary_level | Minimum cases in a cell |
|---|---:|
| `national_year` | 2,024 |
| `national_broad_age_year` | 233 |
| `national_age_year` | 83 |
| `hse_region_year` | 32 |

Projection rows in the manuscript table are a historical rate applied to a population. They are not observed patient counts. The paper uses projection method M1 and the overall observed rate.

## Records held back

Patient-level Irish Hip Fracture Database records are not in this repository. Access is through the National Office of Clinical Audit under its approved researcher pathway. Ethical approval is University of Limerick Science and Engineering Research Ethics Committee application 2024_07_06_S&E. See `data/RESTRICTED.md`.

The following aggregates stay in the private project because the cells are small enough that length of stay could single out a case:

- Service-pathway rows at region–age–year and region–broad-age–year. The smallest cells contain 2 and 6 cases, with mean, median, and total bed days filled in.
- National age-by-ASA counts and length-of-stay summaries. ASA V cells contain 2, 4, 6, and 10 cases. For the two cases aged 65–69 years, mean and median stay are both 9 days.
- The age-stratified logistic regression table, which repeats those ASA V counts.

## Licence

Code in `src/` is released under the MIT licence (`LICENSE`).

Derived tables and figures are released under CC BY 4.0 (`LICENSE-DATA`). Central Statistics Office tables remain under the CSO CC BY 4.0 licence. The two bed-stock files remain parliamentary releases and should be cited as parliamentary questions 36610/23 and 12141/26.

## Citation

Until the article has a DOI, cite [10.5281/zenodo.23859827](https://doi.org/10.5281/zenodo.23859827). After publication, cite the article and give that Zenodo DOI as the archive of the materials.