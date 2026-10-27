# Kim et al. D0 nuclear morphology analysis

This folder contains a reproducible, image-analysis-only workflow for the four
D0 DAPI images (HA 0, 1, 5 and 10 nM) in the published dataset.

Run with the existing Cellpose environment:

```powershell
& 'C:\Users\dodos\miniforge3\envs\cellpose\python.exe' '.\Kim_D0_nuclear_morphology\run_analysis.py'
```

The workflow segments nuclei, performs image-specific QC, extracts geometry
features, fits one balanced joint PCA/GMM representation, selects the cluster
count from k=2–6, maps clusters back to the source images, and reports sample
composition plus within-image spatial heterogeneity.

Outputs are written to `output/`. Existing masks are reused on reruns.
