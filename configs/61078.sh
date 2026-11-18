# Observation inputs. Paths refer to Setonix. Beam files, centres and intervals are discovered automatically.
OBS_ID="61078"
FIELD="EMU_1124-28"
WORK_DIR="/scratch/askaprt/ccordun/time_dep/61078/data"
INPUT_DIR="/scratch/askaprt/ccordun/time_dep/61078/data/61078"
# External reference for Python extraction only; not required by processing.
RACS_FILE="/scratch/askaprt/ccordun/time_dep/RACS-low3_flint_skymodel.fits.gz"

# Imaging and calibration settings.
PB_IMAGE="/askapbuffer/payne/askap-beams/akpb.iquv.closepack36.54.943MHz.SB86386.taylor.fits"
BIN_SECONDS=900
LAST_BIN=  # Automatically select the last complete interval; optionally set a final bin number.
FULL_BEAM_ARCSEC=15
SNAPSHOT_BEAM_ARCSEC=32

# Setonix software and Slurm settings.
PARTITION="askaprt"
ACCOUNT="askaprt"
QOS="normal"
JOB_TIME="04:00:00"

# Software used on Setonix. Change these only for another installation.
MODULE_PATH="/software/projects/askaprt/modulefiles"
ASKAP_MODULES=(pawseyenv/2025.08 singularity/4.1.0-askap askapdata askapsoft/1.24.0)
METADATA_MODULES=(askappy/2.9.1 askappipeline/2.26.8)
CONVOLVE_MODULE="convolve/3.0.0_ubuntu24"
