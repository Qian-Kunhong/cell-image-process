# Data map for the early cell-quality prediction story

## 1. Research question

Can early, non-terminal nuclear images provide a quantitative phenotype that predicts later differentiation efficiency and final culture quality?

The intended logic is:

> Early live Hoechst image → nuclear morphology, texture and spatial phenotype → early quality score → later differentiation efficiency / function → endpoint staining used for validation and interpretation

## 2. What the current archive can and cannot support

### It can support now

1. Develop and test 2D/3D nuclear segmentation and feature extraction using fixed DAPI images.
2. Describe early nuclear-state differences across culture days and HA schedules.
3. Relate nuclear phenotypes to YAP, OCT4, E-cadherin and F-actin within endpoint imaging panels.
4. Define later cell-quality targets from ALB/AFP flow cytometry, albumin secretion, urea secretion, cell density and differentiation-marker images.
5. Estimate within-run and between-run variability for prospective sample-size and model-design decisions.

### It cannot yet support directly

The archive does not currently contain a confirmed table linking an early live-Hoechst image from one well to a later differentiation result from the same well. Therefore, it cannot yet provide a defensible same-well early-to-late prediction model.

The present archive should first be used for retrospective method development, target definition and prospective experiment design. The final prediction model requires a paired longitudinal dataset.

## 3. Data organized by role in the story

| Priority | Role in the story | Available data | Location | Analysis unit | What it can answer | Main limitation |
|---|---|---|---|---|---|---|
| P1 | Early nuclear phenotype | Raw 3D confocal OIB stacks: DAPI + E-cadherin + F-actin; samples D20D–D53D | `03_Raw data/Immunostaining/Confocal - E-cad F-actin` | nucleus, field, sample | 3D nuclear morphology, texture and spatial organization across day/treatment | fixed DAPI, not live Hoechst; small sample count |
| P1 | Early nuclear phenotype and state interpretation | Raw 3D confocal OIB stacks: DAPI + YAP; samples D20D–D53D | `03_Raw data/Immunostaining/Confocal - YAP` | nucleus, field, sample | nuclear phenotype plus YAP nuclear/cytoplasmic state | fixed endpoint; panel is separate from E-cad/F-actin |
| P1 | Existing YAP benchmark | Extracted TIFF channels, CellProfiler object tables and YAP N/C summary | `03_Raw data/Immunostaining/YAP quantification` | cell, image, sample | validate segmentation and reproduce published YAP N/C measurements | present pipeline is mainly 2D |
| P1 | Non-destructive longitudinal reference | Phase-contrast differentiation time-lapse, 97 BMP frames per group HA00/HA01/HA05/HA10 | `03_Raw data/Biostudio/Differentiation` | frame, field, group | early dynamics and later morphological trajectory | not nuclear imaging; pairing to final assays is not yet confirmed |
| P1 | Early culture dynamics | Expansion time-lapse AVI/images for HA0D/HA1D/HA3D | `03_Raw data/Biostudio/Expansion` | frame, field, group | growth, movement and population reorganization before differentiation | not Hoechst; image tiling and identifiers need reconstruction |
| P2 | Early pluripotency/state validation | D0 DAPI/OCT4 IN Cell images at 4× and 10× | `03_Raw data/Immunostaining/IN Cell/D0_OCT34 4x` and `D0_OCT34 10x` | nucleus, field, treatment | link nuclear phenotype to OCT4 status | destructive cross-sectional imaging |
| P2 | Early organization/state validation | D0/D3/D17/D21 DAPI + E-cadherin + N-cadherin IN Cell images | `03_Raw data/Immunostaining/IN Cell/D0_3_17_21_E-cad N-cad 4x` and `10x` | nucleus, field, treatment, day | connect nuclear organization to cell-cell adhesion state | different time points are not the same tracked cells |
| P2 | Early differentiation-state validation | D3 DAPI/SOX2/SOX17 IN Cell images | `03_Raw data/Immunostaining/IN Cell/D3_SOX2 SOX17 4x` and `10x` | nucleus, field, treatment | test whether early nuclear phenotype agrees with early lineage state | destructive cross-sectional imaging |
| P1 | Primary final differentiation target | D17 and D21 ALB/AFP flow cytometry, including FCS and gating files | `03_Raw data/Flow cytometry/Differentiation_AFP-FITC, ALB-PE` | run, sample, replicate | quantify differentiation efficiency, especially ALB+/AFP+ and ALB+ fractions | no confirmed well-level link to early images |
| P1 | Functional final targets | Albumin-secretion plate-reader data and summary | `03_Raw data/Plate reader/Albumin secretion` | run, sample, replicate | secretory function and run variability | no confirmed early-image pairing |
| P1 | Functional final targets | D17/D21 urea-secretion data | `03_Raw data/Plate reader/Urea secretion` | run, sample, replicate | metabolic function and run variability | no confirmed early-image pairing |
| P1 | Integrated final-quality table | Cell density, flow cytometry and secretion summaries for D17/D21 | `03_Raw data/22.07_Analysis of Total cell density_Flow cytometry_Secretion.xlsx` | run, sample, replicate | construct a composite cell-quality target and calculate reproducibility | identifiers must be reconciled with imaging data |
| P2 | Image-based final markers | D21 DAPI/AFP/ALB and DAPI/HNF4α/SOX9 IN Cell images | `03_Raw data/Immunostaining/IN Cell/D21_AFP ALB` and `D21_HNF4a SOX9 4x/10x` | nucleus, field, treatment | spatial/image validation of the final differentiated state | fixed endpoint; not same-cell longitudinal data |
| P2 | Molecular endpoint | D0/D3/D17/D21 qPCR summary and raw EDS files | `03_Raw data/PCR/Hepatocyte differentiation` | treatment, day, replicate | molecular validation of pluripotency and hepatocyte markers | mainly group-level/cross-sectional |
| P2 | Hoechst safety-study reference | Cell counts and live-cell counts for 0D/1D/3D HA conditions | `03_Raw data/Cell growth/Cell count D0_0D 1D 3D HA.xlsx` | treatment, replicate | baseline variability and readout design for future Hoechst safety test | this is not a Hoechst-dose safety experiment |
| P3 | Mechanistic support | E-cadherin, integrin, pMLC and YAP western blots | `03_Raw data/Western blot` | treatment, time | supporting pathway interpretation | not suitable as the primary prediction target |

## 4. Recommended target hierarchy

### Primary prediction target

Use a sample/run-level differentiation-efficiency variable derived from flow cytometry:

- ALB+/AFP+ fraction
- total ALB+ fraction
- optionally a predefined success/failure threshold

### Secondary functional targets

- Albumin secretion
- Urea secretion
- D17/D21 cell density or yield
- Within-run and between-run coefficient of variation

### Interpretation targets, not the main prediction target

- YAP nuclear/cytoplasmic ratio
- OCT4 intensity or positive fraction
- E-cadherin/F-actin organization
- SOX2/SOX17, HNF4α/SOX9 and AFP/ALB image signals

The distinction is important: YAP and OCT4 explain what the early phenotype may represent, while flow cytometry and functional assays define whether the final culture is successful.

## 5. Recommended retrospective analysis with the existing archive

### Stage A — Build the nuclear phenotype space

Use the two raw confocal panels to segment DAPI in 3D and calculate:

- nuclear volume, surface area, sphericity and elongation
- 3D intensity distribution and texture
- local nuclear density and nearest-neighbour distances
- spatial clustering and population composition
- sample-level averages, distributions and heterogeneity

All nuclei from all samples should enter one common feature space before clustering. Each nucleus is one feature vector; fields and samples are summarized only after cell-level clustering.

### Stage B — Test biological interpretability

Within each compatible staining panel, test whether nuclear states are associated with:

- YAP N/C ratio
- E-cadherin organization
- F-actin organization
- OCT4 status in the IN Cell dataset

This stage validates that the nuclear phenotype reflects measurable cell state. It does not yet demonstrate later prediction.

### Stage C — Define a stable final-quality label

Use D17/D21 flow cytometry, albumin, urea and cell-density tables to compare possible targets. Select one primary target before training the final model. Keep functional measures as secondary outcomes.

### Stage D — Prospective paired experiment

Acquire early low-dose live-Hoechst images and preserve the identity of each well/clone. Continue culturing the same unit and collect its later outcome. The minimum metadata table should contain:

| experiment_id | run_id | plate_id | well_id / clone_id | early_day | image_id | Hoechst condition | culture condition | endpoint_day | FCM outcome | albumin | urea | endpoint image IDs |
|---|---|---|---|---|---|---|---|---|---|---|---|---|

The train/test split must be performed by independent run or batch, not by individual nuclei or fields, to prevent information leakage.

## 6. Practical first deliverable

The strongest first result that can be produced from the existing archive is:

> A common 3D nuclear-morphology map across D−3 to D0 and HA schedules, followed by sample-level comparison of nuclear-state composition and heterogeneity, with YAP and E-cadherin/F-actin used for interpretation.

This is the method-development and biological-validation part of the story. Direct early prediction of later differentiation efficiency should be reserved for the prospective paired Hoechst experiment unless reliable sample pairing can be reconstructed.

## 7. Sample-code interpretation used for the confocal stacks

Temporary code format:

> D[culture day][treatment-schedule code]D

- First digit: culture day after seeding; culture day 2/3/4/5 corresponds to paper Day −3/−2/−1/0.
- Second digit: treatment schedule; 0 = control, 1 = single HA schedule, 3 = repeated HA schedule.
- Initial and final D: fixed naming characters; no evidence currently indicates that the last D encodes another condition.
