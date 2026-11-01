# Patient-level Irish Hip Fracture Database records

This directory contains no data files.

Patient-level records from the Irish Hip Fracture Database (IHFD) are not part of this repository. The tables in `data/derived/` are aggregates already used in the manuscript. They are not a substitute for the IHFD extract, and they cannot be turned back into patient rows.

## How to request access

IHFD is maintained by the National Office of Clinical Audit (NOCA). A new analysis of the patient-level records has to go through NOCA's approved researcher pathway. The current route is published by NOCA at <https://www.noca.ie/>.

The extract used for this study was released under that pathway. Ethical approval for this study was given by the University of Limerick Science and Engineering Research Ethics Committee, application 2024_07_06_S&E. That approval covers this study. It does not transfer to another project. A new user needs their own NOCA approval and their own ethics review.

The authors cannot send the patient-level file by email, and it will not be added to this repository or to Zenodo.

## What stays in the private project

Some IHFD aggregates were also held back, because the cells are small enough that a length of stay could single out a case. Those files are not stored in this repository. They include region-by-age-by-year service-pathway rows, national age-by-ASA counts and length-of-stay summaries, and the age-stratified logistic regression table. The manuscript reports those sparse strata only in suppressed form.

A patient-level rebuild of the manuscript tables requires the NOCA extract. The code in this repository checks the released aggregates only.