#!/bin/bash -l
# Start with bash workflow.sbatch config.sh [--dry-run]. Dry runs never call sbatch.
set -eo pipefail

if [[ ${1:-} == --stage ]]; then
    stage=$2
    run=$3
    bundle=${run%/*}
    source "$bundle/config.sh"
    shift 3
else
    [[ $# == 1 || ( $# == 2 && "$2" == --dry-run ) ]] || { echo "Usage: bash workflow.sbatch config.sh [--dry-run]" >&2; exit 2; }
    dry_run=false
    [[ ${2:-} != --dry-run ]] || dry_run=true
    source "$(realpath "$1")"
fi

input=$INPUT_DIR
prefix="scienceData.${FIELD}.SB${OBS_ID}.${FIELD}"
pb=$PB_IMAGE
RESULTS_DIR="$WORK_DIR/solutions/$OBS_ID"

input_ms() {
    awk -v beam="$1" '$1 == beam {$1=""; sub(/^ /, ""); print; found=1} END {if (!found) exit 1}' "$bundle/measurement-sets.txt"
}

# Use the module-provided Python and packages for metadata and analysis.
load_askap_modules() {
    unset PYTHON_PACKAGES PYTHONPATH PYTHONHOME
    export PYTHONNOUSERSITE=1
    module use "$MODULE_PATH"
    module load "${ASKAP_MODULES[0]}"
    export MYSOFTWARE="${MYSOFTWARE:-/software/projects/${PAWSEY_PROJECT:-$ACCOUNT}/$USER}"
    module load "${ASKAP_MODULES[@]:1}" "${METADATA_MODULES[@]}"
}

metadata_python() {
    python "$@"
}

pointing() {
    awk -v beam="$1" '$1 == beam {$1=""; sub(/^ /, ""); print; found=1} END {if (!found) exit 1}' "$bundle/pointings.txt"
}

submit() {
    local stage=$1 nodes=$2 dependency=$3 job time=$JOB_TIME
    [[ -z ${plan_barrier:-} ]] || dependency="${dependency:+$dependency:}$plan_barrier"
    shift 3
    [[ "$stage" != handoff ]] || time=00:15:00
    [[ "$stage" != export ]] || time=01:00:00
    if [[ ${dry_run:-false} == true ]]; then
        local counter
        read -r counter < "$bundle/next-job.txt"
        printf -v job 'DRY%05d' "$counter"
        printf '%s\n' "$((counter + 1))" > "$bundle/next-job.txt"
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$job" "$stage" "$nodes" "$time" "$dependency" "$run $*" >> "$bundle/job-plan.tsv"
        printf '%s %s %s\n' "$job" "$stage" "$*" >> "$run/jobs.txt"
        printf '%s\n' "$job"
        return
    fi
    local options=(--mem=0)
    [[ -z "$dependency" ]] || options+=(--dependency="afterok:$dependency" --kill-on-invalid-dep=yes)
    if [[ "$stage" == handoff ]]; then
        time=00:15:00
        options+=(--ntasks=1 --cpus-per-task=1 --mem=1G)
    elif [[ "$stage" == export ]]; then
        options+=(--ntasks=1 --cpus-per-task=1 --mem=4G)
    fi
    job=$(env -u SLURM_JOB_DEPENDENCY -u SBATCH_DEPENDENCY -u SBATCH_CPUS_PER_TASK -u SBATCH_TRES_PER_TASK sbatch --parsable --partition="$PARTITION" --account="$ACCOUNT" --qos="$QOS" --nodes="$nodes" --threads-per-core=1 --time="$time" --job-name="$OBS_ID-$stage" --output="$run/$stage-%j.out" "${options[@]}" "$bundle/workflow.sbatch" --stage "$stage" "$run" "$@")
    job=${job%%;*}
    [[ "$job" =~ ^[0-9]+$ ]] || return 1
    printf '%s %s %s\n' "$job" "$stage" "$*" >> "$run/jobs.txt"
    printf '%s\n' "$job"
}

queue_images() {
    local branch=$1 window=$2 beam pol job mosaic selavy dependency index=0
    shift 2
    local dependencies=("$@") images=() convolved=() mosaics=()

    while read -r beam; do
        images=()
        dependency=${dependencies[index]:-}
        [[ ${#dependencies[@]} != 1 ]] || dependency=${dependencies[0]}
        for pol in i q u v; do
            job=$(submit image 2 "$dependency" "$branch" "$beam" "$pol" "$window")
            images+=("$job")
        done
        job=$(submit convolve 1 "$(IFS=:; echo "${images[*]}")" "$branch" "$beam")
        convolved+=("$job")
        index=$((index + 1))
    done < "$run/beams.txt"

    for pol in i q u v; do
        mosaic=$(submit mosaic 1 "$(IFS=:; echo "${convolved[*]}")" "$branch" "$pol")
        mosaics+=("$mosaic")
    done
    selavy=$(submit selavy 2 "${mosaics[0]}" "$branch")
    printf '%s:%s\n' "$selavy" "$(IFS=:; echo "${mosaics[*]}")"
}

queue_bin() {
    local index=$1 total bin window beam before_tail after_tail job
    local calibration=()
    total=$(wc -l < "$run/windows.txt")
    (( index >= 1 && index <= total )) || return 1
    read -r bin window < <(sed -n "${index}p" "$run/windows.txt")
    mkdir -- "$run/$bin"

    before_tail=$(queue_images "$bin/before" "$window")
    while read -r beam; do
        job=$(submit cal 1 "$before_tail" "$beam" "$bin" "$window")
        calibration+=("$job")
    done < "$run/beams.txt"
    after_tail=$(queue_images "$bin/after" "$window" "${calibration[@]}")
    echo "$bin queued; final jobs: $after_tail"

    if (( index < total )); then
        job=$(submit handoff 1 "$after_tail" "$((index + 1))")
        echo "Next-bin handoff: $job"
        [[ ${dry_run:-false} != true ]] || plan_barrier="$job"
    fi
}

if [[ ${stage:-} == "" ]]; then
    : "${OBS_ID:?}" "${FIELD:?}" "${INPUT_DIR:?}" "${WORK_DIR:?}" "${PB_IMAGE:?}" "${RESULTS_DIR:?}"
    [[ "$OBS_ID" =~ ^[0-9]+$ && "$FIELD" =~ ^[A-Za-z0-9_.-]+$ ]] || { echo "Use a numeric OBS_ID and a filename-safe FIELD" >&2; exit 2; }
    [[ -n "$PARTITION" && -n "$ACCOUNT" && -n "$QOS" && "$JOB_TIME" =~ ^[0-9]+:[0-5][0-9]:[0-5][0-9]$ ]] || { echo "Check PARTITION, ACCOUNT, QOS and JOB_TIME in the config" >&2; exit 2; }
    scripts_dir=$(cd "$(dirname "$(realpath "$0")")" && pwd)
    [[ -r "$scripts_dir/check_inputs.py" && -r "$scripts_dir/export_pointing.py" ]] || { echo "Missing check_inputs.py or export_pointing.py" >&2; exit 1; }
    load_askap_modules
    MSLIST_PARSER=$(command -v parseMSlistOutput.py || true)
    [[ -n "$MSLIST_PARSER" ]] || MSLIST_PARSER=/software/projects/askaprt/askappipeline/2.26.8/bin/parseMSlistOutput.py
    [[ -r "$MSLIST_PARSER" ]] || { echo "Cannot find parseMSlistOutput.py in the ASKAP pipeline installation" >&2; exit 1; }
    for tool in sbatch srun imager cmodel cbpcalibrator ccalapply linmos-mpi selavy taql mslist; do
        command -v "$tool" > /dev/null || { echo "Missing executable after loading modules: $tool" >&2; exit 1; }
    done
    (
        module load "$CONVOLVE_MODULE"
        command -v beamcon_2D > /dev/null || { echo "Missing beamcon_2D after loading $CONVOLVE_MODULE" >&2; exit 1; }
    )
    mkdir -p "$WORK_DIR"
    if [[ "$dry_run" == true ]]; then
        bundle=$(mktemp -d "$WORK_DIR/dry-run-XXXXXXXX")
    else
        bundle=$(mktemp -d "$WORK_DIR/workflow-XXXXXXXX")
    fi
    echo "Input checks and workflow files: $bundle"
    checks=(--obs-id "$OBS_ID" --field "$FIELD" --work-dir "$WORK_DIR" --input-dir "$INPUT_DIR" --pb "$PB_IMAGE" --seconds "$BIN_SECONDS" --full-beam "$FULL_BEAM_ARCSEC" --snapshot-beam "$SNAPSHOT_BEAM_ARCSEC" --output "$bundle")
    [[ -z ${LAST_BIN:-} ]] || checks+=(--last-bin "$LAST_BIN")
    metadata_python -c 'import numpy, astropy; from casacore.tables import table'
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 metadata_python "$scripts_dir/check_inputs.py" "${checks[@]}"
    source "$bundle/dataset.sh"
    declare -p RESULTS_DIR >> "$bundle/dataset.sh"
    INTERVALS="$bundle/intervals.csv"
    full="$bundle/full"
    run="$full"
    mkdir "$full" "$bundle/snapshots"
    cp "$(realpath "$0")" "$bundle/workflow.sbatch"
    cp "$scripts_dir/export_pointing.py" "$bundle/export_pointing.py"
    declare -p OBS_ID FIELD FIELD_RA FIELD_DEC INPUT_DIR WORK_DIR RESULTS_DIR PB_IMAGE MSLIST_PARSER NCHAN FULL_BEAM_ARCSEC SNAPSHOT_BEAM_ARCSEC PARTITION ACCOUNT QOS JOB_TIME MODULE_PATH ASKAP_MODULES METADATA_MODULES CONVOLVE_MODULE > "$bundle/config.sh"
    for id in {0..35}; do
        printf 'beam%02d\n' "$id" >> "$full/beams.txt"
    done
    cp "$full/beams.txt" "$bundle/snapshots/beams.txt"
    printf '%s\n' "$full" > "$bundle/snapshots/full-workflow.txt"
    if [[ "$dry_run" == true ]]; then
        printf '1\n' > "$bundle/next-job.txt"
        printf 'Job\tStage\tNodes\tTime\tAfter successful jobs\tArguments\n' > "$bundle/job-plan.tsv"
    fi

    # CSV windows are half-open. ASKAP TimeRange endpoints are inclusive.
    metadata_python - "$INTERVALS" "$LAST_BIN" > "$bundle/snapshots/windows.txt" <<'PYWINDOWS'
import csv
import math
import struct
import sys

with open(sys.argv[1], newline="") as handle:
    rows = list(csv.DictReader(handle))
previous_end = None
for index, row in enumerate(rows):
    name = row["folder"]
    start = float(row["start_mjd_seconds_inclusive"])
    end = float(row["end_mjd_seconds_exclusive"])
    if name != f"bin-{index:03d}" or not all(map(math.isfinite, (start, end))) or not 0 <= start < end:
        raise ValueError(f"Invalid interval: {row}")
    if previous_end is not None and start != previous_end:
        raise ValueError(f"Intervals are not contiguous: {name}")
    previous_end = end
    if index <= int(sys.argv[2]):
        bits = struct.unpack(">Q", struct.pack(">d", end))[0]
        inclusive_end = struct.unpack(">d", struct.pack(">Q", bits - 1))[0]
        print(name, f"{start:.9f},{inclusive_end:.9f}")
PYWINDOWS
    [[ -s "$bundle/snapshots/windows.txt" ]]

    pointing_job=$(submit export 1 "")
    echo "Initial pointing export: $pointing_job"
    echo "Full observation: $full"
    before_tail=$(queue_images before "" "$pointing_job")
    calibration=()
    while read -r beam; do
        job=$(submit cal 1 "$before_tail" "$beam")
        calibration+=("$job")
    done < "$full/beams.txt"
    after_tail=$(queue_images after "" "${calibration[@]}")
    echo "Full-observation final jobs: $after_tail"

    run="$bundle/snapshots"
    job=$(submit handoff 1 "$after_tail" 1)
    echo "Snapshots: $run"
    echo "First-bin handoff: $job"
    if [[ "$dry_run" == true ]]; then
        plan_barrier="$job"
        total=$(wc -l < "$run/windows.txt")
        for ((index=1; index<=total; index++)); do
            queue_bin "$index"
        done
        echo "DRY RUN COMPLETE: no jobs submitted; no imaging, calibration, convolution or extraction executed."
        echo "Plan: $bundle/job-plan.tsv (initial pointing export, full track and all selected snapshots)."
        echo "Checks do not validate Slurm account access, sky-model-service access or scientific correctness."
    fi
    echo "Once the required image products are complete, run analyse_measurements.py and extract_sources.py independently."
    echo "Their pointing input will be: $RESULTS_DIR/pointing-$OBS_ID.npz"
    exit
fi

full="$bundle/full"
if [[ "$stage" == handoff ]]; then
    queue_bin "$1"
    exit
fi

load_askap_modules
unset SLURM_CPUS_PER_TASK SLURM_TRES_PER_TASK SRUN_CPUS_PER_TASK

case "$stage" in
export)
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
metadata_python -u "$bundle/export_pointing.py" --obs-id "$OBS_ID" --ms "$(input_ms beam00)" --intervals "$bundle/intervals.csv" --beam-directions "$bundle/pointings.txt" --output "$RESULTS_DIR"
;;
image)
branch=$1
beam=$2
pol=$3
stem="$prefix.${beam}_averaged_cal.leakage"
ms=$(input_ms "$beam")
[[ "$branch" == before || "$branch" == */before ]] || ms="$run/cal/$beam/$stem.bpcal.ms"
[[ -r "$ms/table.dat" ]]
export PMI_NO_FORK=1 PMI_NO_PREINITIALIZE=1
export PMI_MMAP_SYNC_WAIT_TIME=3600
export FI_CXI_RX_MATCH_MODE=hybrid FI_MR_CACHE_MAX_COUNT=0 MPICH_OFI_STARTUP_CONNECT=1
export FI_CXI_DEFAULT_VNI=$(od -vAn -N4 -tu < /dev/urandom)

out="$run/$branch/$beam"
mkdir -p -- "$out"
cd -- "$out"

case "$pol" in
i) cat > "imager.$pol.in" <<EOF
Cimager.barycentre = false
Cimager.calibrate = false
Cimager.combinechannels = true
Cimager.datacolumn = DATA
Cimager.dataset = ${ms}
Cimager.gridder = MPIWProject
Cimager.gridder.MPIWProject.cfrank = 16
Cimager.gridder.MPIWProject.maxsupport = 3072
Cimager.gridder.MPIWProject.nwplanes = 999
Cimager.gridder.MPIWProject.offsetsupport = true
Cimager.gridder.MPIWProject.oversample = 8
Cimager.gridder.MPIWProject.sharecf = true
Cimager.gridder.MPIWProject.variablesupport = true
Cimager.gridder.MPIWProject.wmax = 45000
Cimager.gridder.MPIWProject.wmaxclip = false
Cimager.gridder.snapshotimaging = false
Cimager.gridder.snapshotimaging.clipping = 0.01
Cimager.gridder.snapshotimaging.longtrack = true
Cimager.gridder.snapshotimaging.wtolerance = 2600
Cimager.imagealloc = 
Cimager.Images.cellsize = [2arcsec, 2arcsec]
Cimager.Images.image.i.${FIELD}.SB${OBS_ID}.cont.${FIELD}.$beam.nchan = 1
Cimager.Images.image.i.${FIELD}.SB${OBS_ID}.cont.${FIELD}.$beam.nterms = 2
Cimager.Images.image.i.${FIELD}.SB${OBS_ID}.cont.${FIELD}.$beam.shape = [6144, 6144]
Cimager.Images.Names = ['image.i.${FIELD}.SB${OBS_ID}.cont.${FIELD}.$beam']
Cimager.Images.nyquistgridding = true
Cimager.Images.writeAtMajorCycle = false
Cimager.imagetype = fits
Cimager.nchanpercore = 24
Cimager.ncycles = 15
Cimager.nworkergroups = 3
Cimager.nwriters = 1
Cimager.preconditioner.Names = [Wiener]
Cimager.preconditioner.preservecf = true
Cimager.preconditioner.Wiener.robustness = 0.0
Cimager.restore = true
Cimager.restore.beam = fit
Cimager.restore.beam.cutoff = 0.5
Cimager.restore.preconditioner.Names = [Wiener]
Cimager.restore.preconditioner.preservecf = true
Cimager.restore.preconditioner.suffix = highres
Cimager.restore.preconditioner.Wiener.robustness = -1.0
Cimager.solver = Clean
Cimager.solver.Clean.algorithm = BasisfunctionMFS
Cimager.solver.Clean.decoupled = true
Cimager.solver.Clean.detectdivergence = true
Cimager.solver.Clean.detectmilddivergence = true
Cimager.solver.Clean.divergencelevels = [1.1,2.0,2.0,0.05]
Cimager.solver.Clean.gain = 0.2
Cimager.solver.Clean.logevery = 50
Cimager.solver.Clean.niter = 3000
Cimager.solver.Clean.noiseboxsize = 0
Cimager.solver.Clean.positivity = [false]
Cimager.solver.Clean.psfwidth = 768
Cimager.solver.Clean.scalebias = 1.0
Cimager.solver.Clean.scales = [0,6,15,30,45,60]
Cimager.solver.Clean.solutiontype = MAXBASE
Cimager.solver.Clean.tolerance = 0.01
Cimager.solver.Clean.usecleanmask = false
Cimager.solver.Clean.useincrements = false
Cimager.solver.Clean.usepixellists = false
Cimager.solver.Clean.usepixellists.nsigma = 4.0
Cimager.solver.Clean.verbose = false
Cimager.solver.Clean.weightcutoff = zero
Cimager.solver.Clean.weightcutoff.clean = false
Cimager.solverpercore = false
Cimager.sphfuncforoffset = false
Cimager.threshold.majorcycle = 0.035mJy
Cimager.threshold.masking = 0.9
Cimager.threshold.minorcycle = [30%, 5.0sigma, 0.5sigma]
Cimager.write.firstrestore = true
Cimager.write.maskimage = false
Cimager.write.modelimage = true
Cimager.write.multiple = true
Cimager.write.multiplemodels = true
Cimager.write.psfimage = false
Cimager.write.psfrawimage = false
Cimager.write.residualimage = true
Cimager.write.scalemask = false
Cimager.write.sensitivityimage = false
Cimager.write.weightsimage = false
Cimager.write.weightslog = true
EOF
;;
q) cat > "imager.$pol.in" <<EOF
Cimager.barycentre = false
Cimager.calibrate = false
Cimager.combinechannels = true
Cimager.datacolumn = DATA
Cimager.dataset = $ms
Cimager.gridder = MPIWProject
Cimager.gridder.MPIWProject.cfrank = 16
Cimager.gridder.MPIWProject.maxsupport = 3072
Cimager.gridder.MPIWProject.nwplanes = 999
Cimager.gridder.MPIWProject.offsetsupport = true
Cimager.gridder.MPIWProject.oversample = 8
Cimager.gridder.MPIWProject.sharecf = true
Cimager.gridder.MPIWProject.variablesupport = true
Cimager.gridder.MPIWProject.wmax = 45000
Cimager.gridder.MPIWProject.wmaxclip = false
Cimager.gridder.parotation = true
Cimager.gridder.parotation.angle = 0
Cimager.gridder.snapshotimaging = false
Cimager.gridder.snapshotimaging.clipping = 0.01
Cimager.gridder.snapshotimaging.longtrack = true
Cimager.gridder.snapshotimaging.wtolerance = 2600
Cimager.gridder.swappols = false
Cimager.imagealloc = 
Cimager.Images.cellsize = [2arcsec, 2arcsec]
Cimager.Images.image.q.${FIELD}.SB${OBS_ID}.cont.${FIELD}.$beam.nchan = 1
Cimager.Images.image.q.${FIELD}.SB${OBS_ID}.cont.${FIELD}.$beam.nterms = 2
Cimager.Images.image.q.${FIELD}.SB${OBS_ID}.cont.${FIELD}.$beam.polarisation = ["Q"]
Cimager.Images.Names = [image.q.${FIELD}.SB${OBS_ID}.cont.${FIELD}.$beam]
Cimager.Images.nyquistgridding = true
Cimager.Images.shape = [6144, 6144]
Cimager.Images.writeAtMajorCycle = false
Cimager.imagetype = fits
Cimager.nchanpercore = 24
Cimager.ncycles = 3
Cimager.nworkergroups = 3
Cimager.nwriters = 1
Cimager.preconditioner.Names = [Wiener]
Cimager.preconditioner.preservecf = true
Cimager.preconditioner.Wiener.robustness = 0.0
Cimager.restore = true
Cimager.restore.beam = fit
Cimager.restore.beam.cutoff = 0.5
Cimager.restore.preconditioner.Names = [Wiener]
Cimager.restore.preconditioner.preservecf = true
Cimager.restore.preconditioner.suffix = highres
Cimager.restore.preconditioner.Wiener.robustness = -1.0
Cimager.solver = Clean
Cimager.solver.Clean.algorithm = BasisfunctionMFS
Cimager.solver.Clean.decoupled = true
Cimager.solver.Clean.detectdivergence = true
Cimager.solver.Clean.detectmilddivergence = true
Cimager.solver.Clean.divergencelevels = [1.1,2.0,2.0,0.05]
Cimager.solver.Clean.gain = 0.2
Cimager.solver.Clean.logevery = 50
Cimager.solver.Clean.niter = 800
Cimager.solver.Clean.noiseboxsize = 0
Cimager.solver.Clean.positivity = [false]
Cimager.solver.Clean.psfwidth = 256
Cimager.solver.Clean.scalebias = 1.0
Cimager.solver.Clean.scales = [0,6,15,30,45,60]
Cimager.solver.Clean.solutiontype = MAXBASE
Cimager.solver.Clean.tolerance = 0.01
Cimager.solver.Clean.usecleanmask = false
Cimager.solver.Clean.useincrements = false
Cimager.solver.Clean.usepixellists = false
Cimager.solver.Clean.usepixellists.nsigma = 4.0
Cimager.solver.Clean.verbose = false
Cimager.solver.Clean.weightcutoff = zero
Cimager.solver.Clean.weightcutoff.clean = false
Cimager.solverpercore = false
Cimager.sphfuncforoffset = false
Cimager.threshold.majorcycle = 0.035mJy
Cimager.threshold.masking = 0.9
Cimager.threshold.minorcycle = [30%, 5.0sigma, 0.5sigma]
Cimager.write.firstrestore = true
Cimager.write.maskimage = false
Cimager.write.modelimage = true
Cimager.write.multiple = true
Cimager.write.multiplemodels = true
Cimager.write.psfimage = false
Cimager.write.psfrawimage = false
Cimager.write.residualimage = true
Cimager.write.sensitivityimage = false
Cimager.write.weightsimage = false
Cimager.write.weightslog = true
EOF
;;
u) cat > "imager.$pol.in" <<EOF
Cimager.barycentre = false
Cimager.calibrate = false
Cimager.combinechannels = true
Cimager.datacolumn = DATA
Cimager.dataset = $ms
Cimager.gridder = MPIWProject
Cimager.gridder.MPIWProject.cfrank = 16
Cimager.gridder.MPIWProject.maxsupport = 3072
Cimager.gridder.MPIWProject.nwplanes = 999
Cimager.gridder.MPIWProject.offsetsupport = true
Cimager.gridder.MPIWProject.oversample = 8
Cimager.gridder.MPIWProject.sharecf = true
Cimager.gridder.MPIWProject.variablesupport = true
Cimager.gridder.MPIWProject.wmax = 45000
Cimager.gridder.MPIWProject.wmaxclip = false
Cimager.gridder.parotation = true
Cimager.gridder.parotation.angle = 0
Cimager.gridder.snapshotimaging = false
Cimager.gridder.snapshotimaging.clipping = 0.01
Cimager.gridder.snapshotimaging.longtrack = true
Cimager.gridder.snapshotimaging.wtolerance = 2600
Cimager.gridder.swappols = false
Cimager.imagealloc = 
Cimager.Images.cellsize = [2arcsec, 2arcsec]
Cimager.Images.image.u.${FIELD}.SB${OBS_ID}.cont.${FIELD}.$beam.nchan = 1
Cimager.Images.image.u.${FIELD}.SB${OBS_ID}.cont.${FIELD}.$beam.nterms = 2
Cimager.Images.image.u.${FIELD}.SB${OBS_ID}.cont.${FIELD}.$beam.polarisation = ["U"]
Cimager.Images.Names = [image.u.${FIELD}.SB${OBS_ID}.cont.${FIELD}.$beam]
Cimager.Images.nyquistgridding = true
Cimager.Images.shape = [6144, 6144]
Cimager.Images.writeAtMajorCycle = false
Cimager.imagetype = fits
Cimager.nchanpercore = 24
Cimager.ncycles = 3
Cimager.nworkergroups = 3
Cimager.nwriters = 1
Cimager.preconditioner.Names = [Wiener]
Cimager.preconditioner.preservecf = true
Cimager.preconditioner.Wiener.robustness = 0.0
Cimager.restore = true
Cimager.restore.beam = fit
Cimager.restore.beam.cutoff = 0.5
Cimager.restore.preconditioner.Names = [Wiener]
Cimager.restore.preconditioner.preservecf = true
Cimager.restore.preconditioner.suffix = highres
Cimager.restore.preconditioner.Wiener.robustness = -1.0
Cimager.solver = Clean
Cimager.solver.Clean.algorithm = BasisfunctionMFS
Cimager.solver.Clean.decoupled = true
Cimager.solver.Clean.detectdivergence = true
Cimager.solver.Clean.detectmilddivergence = true
Cimager.solver.Clean.divergencelevels = [1.1,2.0,2.0,0.05]
Cimager.solver.Clean.gain = 0.2
Cimager.solver.Clean.logevery = 50
Cimager.solver.Clean.niter = 800
Cimager.solver.Clean.noiseboxsize = 0
Cimager.solver.Clean.positivity = [false]
Cimager.solver.Clean.psfwidth = 256
Cimager.solver.Clean.scalebias = 1.0
Cimager.solver.Clean.scales = [0,6,15,30,45,60]
Cimager.solver.Clean.solutiontype = MAXBASE
Cimager.solver.Clean.tolerance = 0.01
Cimager.solver.Clean.usecleanmask = false
Cimager.solver.Clean.useincrements = false
Cimager.solver.Clean.usepixellists = false
Cimager.solver.Clean.usepixellists.nsigma = 4.0
Cimager.solver.Clean.verbose = false
Cimager.solver.Clean.weightcutoff = zero
Cimager.solver.Clean.weightcutoff.clean = false
Cimager.solverpercore = false
Cimager.sphfuncforoffset = false
Cimager.threshold.majorcycle = 0.035mJy
Cimager.threshold.masking = 0.9
Cimager.threshold.minorcycle = [30%, 5.0sigma, 0.5sigma]
Cimager.write.firstrestore = true
Cimager.write.maskimage = false
Cimager.write.modelimage = true
Cimager.write.multiple = true
Cimager.write.multiplemodels = true
Cimager.write.psfimage = false
Cimager.write.psfrawimage = false
Cimager.write.residualimage = true
Cimager.write.sensitivityimage = false
Cimager.write.weightsimage = false
Cimager.write.weightslog = true
EOF
;;
v) cat > "imager.$pol.in" <<EOF
Cimager.barycentre = false
Cimager.calibrate = false
Cimager.combinechannels = true
Cimager.datacolumn = DATA
Cimager.dataset = $ms
Cimager.gridder = MPIWProject
Cimager.gridder.MPIWProject.cfrank = 16
Cimager.gridder.MPIWProject.maxsupport = 3072
Cimager.gridder.MPIWProject.nwplanes = 999
Cimager.gridder.MPIWProject.offsetsupport = true
Cimager.gridder.MPIWProject.oversample = 8
Cimager.gridder.MPIWProject.sharecf = true
Cimager.gridder.MPIWProject.variablesupport = true
Cimager.gridder.MPIWProject.wmax = 45000
Cimager.gridder.MPIWProject.wmaxclip = false
Cimager.gridder.parotation = true
Cimager.gridder.parotation.angle = 0
Cimager.gridder.snapshotimaging = false
Cimager.gridder.snapshotimaging.clipping = 0.01
Cimager.gridder.snapshotimaging.longtrack = true
Cimager.gridder.snapshotimaging.wtolerance = 2600
Cimager.gridder.swappols = false
Cimager.imagealloc = 
Cimager.Images.cellsize = [2arcsec, 2arcsec]
Cimager.Images.image.v.${FIELD}.SB${OBS_ID}.cont.${FIELD}.$beam.nchan = 1
Cimager.Images.image.v.${FIELD}.SB${OBS_ID}.cont.${FIELD}.$beam.nterms = 2
Cimager.Images.image.v.${FIELD}.SB${OBS_ID}.cont.${FIELD}.$beam.polarisation = ["V"]
Cimager.Images.Names = [image.v.${FIELD}.SB${OBS_ID}.cont.${FIELD}.$beam]
Cimager.Images.nyquistgridding = true
Cimager.Images.shape = [6144, 6144]
Cimager.Images.writeAtMajorCycle = false
Cimager.imagetype = fits
Cimager.nchanpercore = 24
Cimager.ncycles = 3
Cimager.nworkergroups = 3
Cimager.nwriters = 1
Cimager.preconditioner.Names = [Wiener]
Cimager.preconditioner.preservecf = true
Cimager.preconditioner.Wiener.robustness = 0.0
Cimager.restore = true
Cimager.restore.beam = fit
Cimager.restore.beam.cutoff = 0.5
Cimager.restore.preconditioner.Names = [Wiener]
Cimager.restore.preconditioner.preservecf = true
Cimager.restore.preconditioner.suffix = highres
Cimager.restore.preconditioner.Wiener.robustness = -1.0
Cimager.solver = Clean
Cimager.solver.Clean.algorithm = BasisfunctionMFS
Cimager.solver.Clean.decoupled = true
Cimager.solver.Clean.detectdivergence = true
Cimager.solver.Clean.detectmilddivergence = true
Cimager.solver.Clean.divergencelevels = [1.1,2.0,2.0,0.05]
Cimager.solver.Clean.gain = 0.2
Cimager.solver.Clean.logevery = 50
Cimager.solver.Clean.niter = 800
Cimager.solver.Clean.noiseboxsize = 0
Cimager.solver.Clean.positivity = [false]
Cimager.solver.Clean.psfwidth = 256
Cimager.solver.Clean.scalebias = 1.0
Cimager.solver.Clean.scales = [0,6,15,30,45,60]
Cimager.solver.Clean.solutiontype = MAXBASE
Cimager.solver.Clean.tolerance = 0.01
Cimager.solver.Clean.usecleanmask = false
Cimager.solver.Clean.useincrements = false
Cimager.solver.Clean.usepixellists = false
Cimager.solver.Clean.usepixellists.nsigma = 4.0
Cimager.solver.Clean.verbose = false
Cimager.solver.Clean.weightcutoff = zero
Cimager.solver.Clean.weightcutoff.clean = false
Cimager.solverpercore = false
Cimager.sphfuncforoffset = false
Cimager.threshold.majorcycle = 0.035mJy
Cimager.threshold.masking = 0.9
Cimager.threshold.minorcycle = [30%, 5.0sigma, 0.5sigma]
Cimager.write.firstrestore = true
Cimager.write.maskimage = false
Cimager.write.modelimage = true
Cimager.write.multiple = true
Cimager.write.multiplemodels = true
Cimager.write.psfimage = false
Cimager.write.psfrawimage = false
Cimager.write.residualimage = true
Cimager.write.sensitivityimage = false
Cimager.write.weightsimage = false
Cimager.write.weightslog = true
EOF
;;
    esac
[[ -z "${4:-}" ]] || printf 'Cimager.TimeRange = [%s]\n' "$4" >> "imager.$pol.in"
srun --export=ALL,OMP_NUM_THREADS=16 --cpus-per-task=16 --ntasks=1 imager -c "imager.$pol.in" : --export=ALL,OMP_NUM_THREADS=3 --cpus-per-task=3 --ntasks=36 --ntasks-per-node=36 imager -c "imager.$pol.in" > "imager.$pol.log" 2>&1



;;
cal)
if [[ -n "${2:-}" ]]; then
beam=$1
bin=$2
window=$3
stem="$prefix.${beam}_averaged_cal.leakage"
source_ms=$(input_ms "$beam")
working="$run/cal/$beam/$stem.bpcal.ms"
out="$run/$bin/cal/$beam"
mkdir -p -- "${out%/*}" "${working%/*}"
mkdir -- "$out"
cd -- "$out"

# Reuse the full-track beam model and solve against the original data in this bin.
full_cal="$full/cal/$beam"
model_base="$full_cal/contmodel.i.${FIELD}.SB${OBS_ID}.cont.${FIELD}.$beam"
awk -v model="$model_base" -v window="$window" '
    /^Cbpcalibrator\.sources\.lsm\.model[[:space:]]*=/ {
        print "Cbpcalibrator.sources.lsm.model = " model
        next
    }
    {print}
    END {print "Cbpcalibrator.TimeRange = [" window "]"}
' "$full_cal/cbpcal.in" > cbpcal.in

srun --export=ALL,OMP_NUM_THREADS=1 --ntasks=1 --cpus-per-task=1 cbpcalibrator -c cbpcal.in > cbpcal.log 2>&1

solution="$PWD/$beam.bandpass.tab"
[[ -r "$solution/table.dat" ]]
if [[ ! -e "$working" ]]; then
    cp -rL -- "$source_ms" "$working.partial"
    mv -- "$working.partial" "$working"
fi

cat > ccalapply.in <<PARSET
Ccalapply.dataset = $working
Ccalapply.TimeRange = [$window]
Ccalapply.datacolumn = DATA
Ccalapply.calibrate.allowflag = true
Ccalapply.calibrate.scalenoise = false
Ccalapply.calibrate.ignoreleakage = true
Ccalapply.calibaccess = table
Ccalapply.calibaccess.table = $solution
Ccalapply.calibaccess.table.maxant = 36
Ccalapply.calibaccess.table.maxbeam = 36
Ccalapply.calibaccess.table.maxchan = $NCHAN
PARSET

srun --export=ALL --ntasks=1 --cpus-per-task=1 ccalapply -c ccalapply.in > ccalapply.log 2>&1
echo "Applied $bin [$window] to $working"
else
beam=$1
beam_id=$((10#${beam#beam}))
stem="$prefix.${beam}_averaged_cal.leakage"
source_ms=$(input_ms "$beam")
ms="$source_ms"
nchan=$NCHAN
out="$run/cal/$beam"
mkdir -p -- "$out"
cd -- "$out"

# SB${OBS_ID} reference pipeline pointing and beam metadata.
modelDirection=$(pointing "$beam")
model_base=contmodel.i.${FIELD}.SB${OBS_ID}.cont.${FIELD}.$beam
metadata="$out/mslist-$stem.txt"
srun --export=ALL --ntasks=1 --cpus-per-task=1 mslist --full "$ms" > "$metadata" 2>&1
parser=$MSLIST_PARSER
centreFreq=$(metadata_python "$parser" --file="$metadata" --val=Freq)
bandwidth=$(metadata_python "$parser" --file="$metadata" --val=Bandwidth)
for value in "$centreFreq" "$bandwidth"; do
    [[ "$value" =~ ^[0-9]+([.][0-9]+)?([eE][+-]?[0-9]+)?$ ]] || { echo "Invalid frequency metadata: $value" >&2; exit 1; }
done

# Build each beam's full-track RACS-global model.
cat > cmodel.in <<EOF
Cmodel.bunit              = Jy/pixel
Cmodel.frequency          = ${centreFreq}Hz
Cmodel.increment          = ${bandwidth}Hz
# flux limit for cmodel determined from image noise - see below
Cmodel.shape              = [6144,6144]
Cmodel.cellsize           = [2arcsec, 2arcsec]
Cmodel.direction          = ${modelDirection}
Cmodel.stokes             = [I]
Cmodel.nterms             = 2
Cmodel.nearest            = true

# Output specific parameters
Cmodel.output             = casa
Cmodel.filename           = $model_base


Cmodel.gsm.database          = dataservice
Cmodel.gsm.locator_host      = askap-ingest-service02.pawsey.org.au
Cmodel.gsm.locator_port      = 4061
Cmodel.gsm.service_name      = SkyModelService
Cmodel.gsm.adapter_name      = SkyModelServiceAdaptor_racs_global
Cmodel.gsm.ref_freq          = 943.49MHz
Cmodel.forceSpectralProperty = true


# Using a Taylor term beam FITS image derived from holography measurements
Cmodel.primarybeam                  = ASKAP_PB
Cmodel.primarybeam.ASKAP_PB.beam_number = $beam_id
Cmodel.primarybeam.ASKAP_PB.image   = $pb
Cmodel.primarybeam.ASKAP_PB.nterms  = 3
Cmodel.primarybeam.ASKAP_PB.alpha   = 0 # In radians

Cmodel.flux_limit = 0mJy
EOF

srun --export=ALL,OMP_NUM_THREADS=1 --ntasks=2 --ntasks-per-node=2 --cpus-per-task=1 cmodel -c cmodel.in > cmodel.log 2>&1
for term in 0 1; do
    [[ -r "$model_base.taylor.$term/table.dat" ]] || { echo "Missing Taylor model $term. See cmodel.log" >&2; exit 1; }
done

cat > cbpcal.in <<EOF
Cbpcalibrator.dataset = $ms
Cbpcalibrator.datacolumn = DATA
Cbpcalibrator.nAnt = 36
Cbpcalibrator.nBeam = 1
Cbpcalibrator.beamindices = [$beam_id]
Cbpcalibrator.nChan = $nchan
Cbpcalibrator.solve = bandpass
Cbpcalibrator.solver = SVD
Cbpcalibrator.ncycles = 25
Cbpcalibrator.imagetype = casa
Cbpcalibrator.sources.names = [lsm]
Cbpcalibrator.sources.lsm.direction = $modelDirection
Cbpcalibrator.sources.lsm.model = $model_base
Cbpcalibrator.sources.lsm.nterms = 2
Cbpcalibrator.visweights = MFS
Cbpcalibrator.visweights.MFS.reffreq = $centreFreq
Cbpcalibrator.gridder = WProject
Cbpcalibrator.gridder.WProject.wmax = 45000
Cbpcalibrator.gridder.WProject.wmaxclip = false
Cbpcalibrator.gridder.WProject.nwplanes = 999
Cbpcalibrator.gridder.WProject.oversample = 8
Cbpcalibrator.gridder.WProject.maxsupport = 3072
Cbpcalibrator.gridder.WProject.variablesupport = true
Cbpcalibrator.gridder.WProject.offsetsupport = true
Cbpcalibrator.calibaccess = table
Cbpcalibrator.calibaccess.table = $beam.bandpass.tab
Cbpcalibrator.calibaccess.table.maxant = 36
Cbpcalibrator.calibaccess.table.maxbeam = 36
Cbpcalibrator.calibaccess.table.maxchan = $nchan
Cbpcalibrator.calibaccess.table.reuse = false
EOF

srun --export=ALL,OMP_NUM_THREADS=1 --ntasks=1 --cpus-per-task=1 cbpcalibrator -c cbpcal.in > cbpcal.log 2>&1

solution="$PWD/$beam.bandpass.tab"
[[ -r "$solution/table.dat" ]]
partial="$stem.bpcal.partial.ms"
final="$stem.bpcal.ms"
cp -rL -- "$source_ms" "$partial"

cat > ccalapply.in <<PARSET
Ccalapply.dataset = $partial
Ccalapply.datacolumn = DATA
Ccalapply.calibrate.allowflag = true
Ccalapply.calibrate.scalenoise = false
Ccalapply.calibrate.ignoreleakage = true
Ccalapply.calibaccess = table
Ccalapply.calibaccess.table = $solution
Ccalapply.calibaccess.table.maxant = 36
Ccalapply.calibaccess.table.maxbeam = 36
Ccalapply.calibaccess.table.maxchan = $NCHAN
PARSET

srun --export=ALL --ntasks=1 --cpus-per-task=1 ccalapply -c ccalapply.in > ccalapply.log 2>&1
mv -- "$partial" "$final"
echo "Calibrated MS: $out/$final"
fi
;;

convolve)
branch=$1
beam=$2
cd "$run/$branch/$beam"
beam_size=$FULL_BEAM_ARCSEC
[[ "$branch" != bin-*/* ]] || beam_size=$SNAPSHOT_BEAM_ARCSEC
module load "$CONVOLVE_MODULE"
for pol in i q u v; do
    for term in 0 1; do
        image="image.$pol.${FIELD}.SB${OBS_ID}.cont.${FIELD}.$beam.taylor.$term.restored"
        [[ -s "$image.fits" ]]
        srun --export=ALL --ntasks=1 --cpus-per-task=16 beamcon_2D -s conv -v             --bmaj "$beam_size" --bmin "$beam_size" --bpa 0 --cutoff "$beam_size" --ncores 1             --log "beamLog.$pol.$term.txt" "$image.fits" > "convolve.$pol.$term.log" 2>&1
        [[ -s "$image.conv.fits" ]]
    done
done
;;
mosaic)
branch=$1
pol=$2
cd "$run/$branch"
images=()
weights=()
while read -r beam; do
    stem="$pol.${FIELD}.SB${OBS_ID}.cont.${FIELD}.$beam"
    images+=("./$beam/image.$stem.taylor.0.restored.conv")
    weights+=("./$beam/weights.$stem.taylor.0")
    [[ -s "$beam/weights.$stem.taylor.0.txt" ]]
done < "$run/beams.txt"
names=$(IFS=,; echo "${images[*]}")
wt=$(IFS=,; echo "${weights[*]}")
leakage=true
[[ "$pol" != i ]] || leakage=false
cat > "linmos.$pol.in" <<EOF
linmos.names = [$names]
linmos.weights = [$wt]
linmos.useweightslog = true
linmos.imagetype = fits
linmos.outname = image.$pol.SB${OBS_ID}.linmos.taylor.0.restored.conv
linmos.outweight = weights.$pol.SB${OBS_ID}.linmos.taylor.0.restored.conv
linmos.weighttype = Combined
linmos.weightstate = Inherent
linmos.psfref = 0
linmos.cutoff = 0.2
linmos.primarybeam = ASKAP_PB
linmos.primarybeam.ASKAP_PB.image = $pb
linmos.primarybeam.ASKAP_PB.nterms = 3
linmos.primarybeam.ASKAP_PB.alpha = 0
linmos.removebeam = true
linmos.nterms = 2
linmos.removeleakage = $leakage
linmos.trimming = true
linmos.trimming.type = aggressive
EOF
export OMP_NUM_THREADS=16
srun --export=ALL --ntasks=1 --cpus-per-task=16 linmos-mpi -c "linmos.$pol.in" > "linmos.$pol.log" 2>&1
[[ -s "image.$pol.SB${OBS_ID}.linmos.taylor.0.restored.conv.fits" ]]
;;
selavy)
branch=$1
image="$run/$branch/image.i.SB${OBS_ID}.linmos.taylor.0.restored.conv.fits"
weights="$run/$branch/weights.i.SB${OBS_ID}.linmos.taylor.0.restored.conv.fits"
[[ -s "$image" && -s "$weights" ]]
out="${image%/*}/selavy-$SLURM_JOB_ID"
mkdir -- "$out"
cd -- "$out"

map_suffix=beam00
[[ "$branch" != bin-*/* ]] || map_suffix=mosaic
cat > selavy.in <<PARSET
Selavy.image = $image
Selavy.imagetype = fits
Selavy.sbid = ${OBS_ID}
Selavy.sourceIdBase = SB${OBS_ID}
Selavy.nsubx = 11
Selavy.nsuby = 11
Selavy.overlapx = 0
Selavy.overlapy = 0
Selavy.resultsFile = selavy-SB${OBS_ID}.txt
Selavy.writeDuchampFiles = false
Selavy.Weights.weightsImage = $weights
Selavy.Weights.weightsCutoff = 0.25
Selavy.snrCut = 5
Selavy.flagGrowth = true
Selavy.growthCut = 3
Selavy.VariableThreshold = true
Selavy.VariableThreshold.reuse = false
Selavy.VariableThreshold.boxSize = 50
Selavy.VariableThreshold.ThresholdImageName = detThresh.$map_suffix
Selavy.VariableThreshold.NoiseImageName = noiseMap.$map_suffix
Selavy.VariableThreshold.AverageImageName = meanMap.$map_suffix
Selavy.VariableThreshold.SNRimageName = snrMap.$map_suffix
Selavy.VariableThreshold.imagetype = fits
Selavy.Fitter.doFit = true
Selavy.Fitter.fitTypes = [full]
Selavy.Fitter.numGaussFromGuess = true
Selavy.Fitter.maxReducedChisq = 10.
Selavy.Fitter.imagetype = fits
Selavy.threshSpatial = 5
Selavy.flagAdjacent = true
Selavy.minPix = 3
Selavy.minVoxels = 3
Selavy.minChannels = 1
Selavy.sortingParam = -pflux
PARSET

srun --export=ALL,OMP_NUM_THREADS=1 --ntasks=122 --ntasks-per-node=64 --cpus-per-task=1 selavy -c selavy.in > selavy.log 2>&1
echo "Selavy outputs: $out"

;;
*) echo "Unknown stage: $stage" >&2; exit 2 ;;
esac
