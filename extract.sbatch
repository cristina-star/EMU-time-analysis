#!/bin/bash -l
# Submit from this directory: bash extract.sbatch --submit SCRIPT.py --workflow FOLDER [overrides]
# Lightweight exports: make_intervals.py, export_pointing.py. Measurements: analyse_measurements.py, extract_sources.py.
#SBATCH --partition=askaprt
#SBATCH --account=askaprt
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=analysis-%j.out

set -eo pipefail
scripts_dir=${TIME_DEP_SCRIPTS_DIR:-${SLURM_SUBMIT_DIR:-$PWD}}
local_dir=$(cd "$(dirname "$0")" && pwd)
[[ ! -r "$local_dir/analyse_measurements.py" ]] || scripts_dir=$local_dir
export TIME_DEP_SCRIPTS_DIR="$scripts_dir"
submit=false
if [[ ${1:-} == --submit ]]; then
    submit=true
    shift
fi
config=""
if [[ ${1:-} == --config ]]; then
    config=$(realpath "${2:?Pass the observation config}")
    shift 2
fi
script=${1:?Pass analyse_measurements.py or extract_sources.py and --workflow FOLDER}
shift

MODULE_PATH=/software/projects/askaprt/modulefiles
ASKAP_MODULES=(pawseyenv/2025.08 singularity/4.1.0-askap askapdata askapsoft/1.24.0)
METADATA_MODULES=(askappy/2.9.1 askappipeline/2.26.8)
PARTITION=askaprt
ACCOUNT=askaprt
QOS=normal

workflow=""
arguments=("$@")
for ((i=0; i<${#arguments[@]}; i++)); do
    if [[ ${arguments[i]} == --workflow ]]; then
        workflow=${arguments[i+1]:?Pass the workflow folder}
    elif [[ ${arguments[i]} == --workflow=* ]]; then
        workflow=${arguments[i]#--workflow=}
    fi
done
if [[ -n "$config" ]]; then
    source "$config"
elif [[ -n "$workflow" && -r "$workflow/config.sh" ]]; then
    source "$workflow/config.sh"
    # Read observation defaults before the saved workflow settings.
    if [[ -r "$scripts_dir/configs/$OBS_ID.sh" ]]; then
        source "$scripts_dir/configs/$OBS_ID.sh"
        source "$workflow/config.sh"
    fi
fi

if [[ "$submit" == true ]]; then
    case "$script" in
        analyse_measurements.py|extract_sources.py) ;;
        *) echo "--submit supports analyse_measurements.py or extract_sources.py" >&2; exit 2 ;;
    esac
    options=()
    [[ -z "$config" ]] || options+=(--config "$config")
    logs=${RESULTS_DIR:-$PWD}
    mkdir -p "$logs"
    exec env -u SLURM_JOB_DEPENDENCY -u SBATCH_DEPENDENCY -u SBATCH_CPUS_PER_TASK -u SBATCH_TRES_PER_TASK sbatch --partition="$PARTITION" --account="$ACCOUNT" --qos="$QOS" --nodes=1 --ntasks=1 --cpus-per-task=1 --mem=32G --time=04:00:00 --job-name="${OBS_ID:-time}-${script%.py}" --output="$logs/${script%.py}-%j.out" "$scripts_dir/extract.sbatch" "${options[@]}" "$script" "$@"
fi

case "$script" in
    make_intervals.py|export_pointing.py|analyse_measurements.py|extract_sources.py) ;;
    *) echo "Choose make_intervals.py, export_pointing.py, analyse_measurements.py or extract_sources.py" >&2; exit 2 ;;
esac

# Ignore personal Python settings, including those saved in older workflows.
unset PYTHON_PACKAGES PYTHONPATH PYTHONHOME
export PYTHONNOUSERSITE=1
module use "$MODULE_PATH"
module load "${ASKAP_MODULES[0]}"
export MYSOFTWARE="${MYSOFTWARE:-/software/projects/${PAWSEY_PROJECT:-$ACCOUNT}/$USER}"
module load "${ASKAP_MODULES[@]:1}" "${METADATA_MODULES[@]}"

export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/mpl-analysis-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"

exec python -u "$scripts_dir/$script" "$@"
