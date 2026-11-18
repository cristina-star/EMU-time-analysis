# EMU calibration and imaging analysis


These scripts and notebooks were developed during my visit to CSIRO for work on the **Evolutionary Map of the Universe (EMU) survey with ASKAP**. They support the analyses presented in [ASKAP_cal_notes.pdf](ASKAP_cal_notes.pdf), which describes the methods and results of this work. The analyses examine how calibration and imaging choices affect EMU images and source catalogues, including their variation with elevation.

For the full observation and each 15-minute interval, the workflow first makes images from the pipeline-calibrated data. It then runs an extra bandpass-calibration step with cbpcalibrator, solving for frequency-dependent amplitude and phase gains against a RACS-global sky model for each beam. These gains are applied to a copy of the data, and the imaging is repeated to produce before-and-after comparisons. Two separate Python programs extract image-quality measurements, gain amplitudes and individual source fluxes. The notebooks plot and compare these results.

Here, **before** means before the additional calibration. The input data must already have the standard pipeline calibration applied.

Note: I designed the workflow and wrote the Python extraction code. ChatGPT helped clean up and organise the code, assemble the workflow scripts, and write the README. The workflow and extraction scripts were tested on Setonix and reproduced my existing results.

## What is included?

| File | Purpose |
| --- | --- |
| `workflow.sbatch` | Process the full observation and the time intervals, including calibration, imaging, mosaics and source finding. |
| `analyse_measurements.py` | Measure image noise, flux scale, positional offsets, polarisation leakage, dynamic range and gain amplitudes. |
| `extract_sources.py` | Detect sources in the first interval and track their peak flux densities through the before-calibration images. |
| `extract.sbatch` | Load the analysis environment and submit either Python extractor on Setonix. |
| [`time_dep.ipynb`](postproc_notebooks/time_dep.ipynb) | Plot the time-dependent measurements, gain solutions, source fluxes and antenna pointing angles. |
| [`compare_amp.ipynb`](postproc_notebooks/compare_amp.ipynb) | Compare existing pipeline products made with different amplitude self-calibration settings. |

Keep the scripts, helper files and configs/ together. The notebooks use existing results to make plots.

## 1. Prepare the inputs

Processing runs on **Setonix** using ASKAP software and the Slurm job scheduler. You need:

- Already-calibrated measurement sets (MSs), one for each of the 36 beams. Filenames must contain `beam00` through `beam35`.
- The original pipeline beam-footprint file, `footprintOutput-sbID-srcN-FIELD.txt`, containing the 36 distinct beam centres. Put it in `WORK_DIR/metadata/`; the workflow can also find it in nearby metadata directories.
- A suitable primary-beam FITS model, describing the beams' response on the sky.
- A RACS-low3 reference catalogue for the later Python measurements and image alignment.
- Access to the software modules, Slurm account and sky-model service specified in the config.

The current scripts support 36 beams, 36 antenna entries, 288 channels, one spectral window and XX/XY/YX/YY correlations. The MSs must contain the ASKAP POINTING angle columns.

Copy the folder to Setonix and run the following commands from inside it. Choose an example config:

```bash
config=$(realpath configs/86292.sh)
```

For another observation, copy a config and edit the observation ID, field name, input and working directories, primary-beam model, reference catalogue and Slurm settings. Update software paths/modules if needed. 

The workflow discovers the MS filenames and calculates the time intervals and beam coordinates automatically. In the config:

- `BIN_SECONDS=900` gives 15-minute intervals.
- Leave `LAST_BIN` empty to process all complete intervals. Set it to `1` for a trial with the full observation and bins 000–001.
- `FULL_BEAM_ARCSEC` and `SNAPSHOT_BEAM_ARCSEC` set the common image resolution. The examples use 15 and 32 arcsec, respectively. The snapshot images have poorer resolution because the beam is elongaed due to the small uv coverage.

Calibration uses **RACS-global at 943.49 MHz** through the sky-model service. `RACS_FILE` is a separate reference catalogue used only by the Python extractors.

## 2. Check the setup, then process

First make a dry run, which checks inputs and writes the planned jobs without submitting them:

```bash
bash workflow.sbatch "$config" --dry-run
```

In the printed `dry-run-XXXXXXXX/` folder, check `input-check.json`, `pointings.txt`, `intervals.csv` and `job-plan.tsv`. Confirm that the footprint belongs to this observation and has 36 different beam centres. A dry run checks the setup; a small real run is still needed to check the resulting images.

Start processing with:

```bash
bash workflow.sbatch "$config"
```

The launcher prints a new `workflow-XXXXXXXX/` folder. It exports the pointing metadata first, processes the full observation, then processes each time interval in order. Within each stage, beams and Stokes I/Q/U/V images can run in parallel as resources allow. The next stage waits for its required jobs to finish successfully.

Each full-observation or interval run performs before imaging, additional calibration, and after imaging. Both image sets are convolved to the chosen resolution and combined into mosaics, followed by Selavy source finding on Stokes I. Gains are applied to working copies of the MSs. Snapshots reuse the full-observation RACS-global models. No imaging UV cut is applied.

The workflow saves its settings, input checks and intervals alongside the products:

```text
workflow-XXXXXXXX/
    full/before/ and full/after/
    snapshots/bin-000/before/ and snapshots/bin-000/after/
    snapshots/bin-001/...
```

## 3. Extract the measurements

Extraction is a separate step; it does not rerun imaging or calibration. Set the workflow folder printed by the launcher:

```bash
bundle=/absolute/path/to/workflow-XXXXXXXX
bash extract.sbatch --submit analyse_measurements.py --workflow "$bundle"
bash extract.sbatch --submit extract_sources.py --workflow "$bundle"
```

These two jobs can run independently and in parallel. Each requests one CPU, 32 GB and four hours. Measurement extraction needs the completed full-observation before/after products and uses complete snapshot bins. Source extraction needs the reference-bin before images and at least two available bins.

The source extractor selects image peaks above 10 mJy/beam in bin-000. It aligns images to the reference catalogue and measures the maximum finite pixel within 10 arcsec of each fixed source position in later bins. The brightness cut is applied only in the reference bin. Missing measurements are saved as NaN.

Results and job logs go to `WORK_DIR/solutions/OBS_ID/`. To check the inferred input paths without submitting a job:

```bash
bash extract.sbatch analyse_measurements.py --workflow "$bundle" --show-inputs
bash extract.sbatch extract_sources.py --workflow "$bundle" --show-inputs
```

Use `--output /path/to/test-results` for a separate output folder. The extractors also provide `--help` for options such as source thresholds, selected bins and resuming measurements.

## 4. View the results in the notebooks

Use Jupyter with NumPy, Matplotlib and Astropy. Run each section's function definitions before its example calls, and update the data and figure paths for your machine.

### Time-dependent analysis

For `time_dep.ipynb`, copy these three files from the results directory to your laptop. For example:

```text
time_dep/
    solutions/86292/
        measurements.npz
        SB86292_all_beam_source_peaks_bins000-039.npz
        SB86292_pointing_angles.npz
```

The source filename reflects the available bin range. Keep one active all-source archive per observation. `pointing-ID.npz` is an intermediate file needed by the extractors, not the notebook.

In the notebook, set your local folder and load an observation:

```python
base = Path("/path/to/time_dep")
sb86292 = load_time_data("86292", base=base)
```

Then run the plotting calls for that observation. Figures are saved in `plots/OBS_ID/`. The optional image-comparison cells also need the FITS images at the paths specified in those cells. The examples cover SB86292 and SB61078.

### Amplitude self-calibration comparison

`compare_amp.ipynb` compares existing runs with different amplitude self-calibration settings. Its examples use SB87627. Supply the calibration tables (`caldata_loop*.tab`), Selavy component catalogues, RACS-low3 catalogue and FITS images for the sections you want to run, and edit the paths and selected cases in those sections. Reading the calibration tables also requires the casacore `taql` command.
