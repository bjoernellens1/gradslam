#!/usr/bin/env bash
# Run the full TUM + Orbbec FPS benchmark matrix.
# Usage: ./run_benchmark_matrix.sh <host_tag> <out_root>
set -euo pipefail

HOST="${1:-unknown}"
OUT_ROOT="${2:-/workspace/outputs/benchmarks/${HOST}}"
TUM_ROOT=/workspace/datasets/public/TUM/tum_rgbd
# The bjoern/orbbec-handheld/ mcap is truncated; the christian/embedding_map
# slam bags have valid MCAP footers and the same /camera/{color,depth} topics.
ORBBEC_BAG="${ORBBEC_BAG:-/workspace/datasets/christian/embedding_map/bags/slam/kitchen1_slam/kitchen1_slam_0.mcap}"
ORBBEC_DIR="${OUT_ROOT}/orbbec_extracted"
# fr2_large_no_loop is shipped as a .tgz on the read-only share. The runner
# extracts it once into OUT_ROOT (writable), then points TUM_ROOT at the
# extracted location for that one sequence.
TUM_ROOT_FR2="${OUT_ROOT}/tum_freiburg2_root"

cd /workspace/gradslam
mkdir -p "${OUT_ROOT}"

echo "==[${HOST}]== Ensure rosbags is installed (no-op if already there)"
python -c "import rosbags" 2>/dev/null || pip install --quiet "rosbags>=0.10"

echo "==[${HOST}]== ROCm stack check"
python scripts/check_rocm_stack.py 2>&1 | tail -20 || true

echo "==[${HOST}]== Extract Orbbec mcap (if not already)"
if [ ! -f "${ORBBEC_DIR}/camera_info.json" ]; then
  python scripts/extract_mcap_orbbec.py "${ORBBEC_BAG}" "${ORBBEC_DIR}"
else
  echo "  already extracted: $(ls ${ORBBEC_DIR}/images | wc -l) frames"
fi

run_slam() {
  local ds="$1" seq="$2" backend="$3" mode="$4" extra="$5" outdir="$6" cap="$7"
  echo "==[${HOST}]== SLAM ${ds}/${seq}  backend=${backend}  mode=${mode}  extra='${extra}'  cap=${cap:-0}"
  local cmd=(python scripts/run_slam.py "${ds}")
  if [ "${ds}" = "tum" ]; then
    cmd+=( --dataset-root "${TUM_ROOT}" --sequence "${seq}" )
  else
    cmd+=( --capture-dir "${ORBBEC_DIR}" )
  fi
  cmd+=( --slam-backend "${backend}" --tracking-mode "${mode}" --output "${outdir}" )
  if [ -n "${extra}" ]; then
    # shellcheck disable=SC2206
    extra_arr=( ${extra} )
    cmd+=( "${extra_arr[@]}" )
  fi
  if [ -n "${cap}" ] && [ "${cap}" -gt 0 ]; then
    cmd+=( --max-frames "${cap}" )
  fi
  "${cmd[@]}" 2>&1 | tee "${outdir}.log" | tail -25 || true
}

for SEQ in freiburg1_xyz freiburg1_desk; do
  out="${OUT_ROOT}/tum_${SEQ}_speed"
  mkdir -p "${out}"
  run_slam tum "${SEQ}" fast_rgbd fast_rgbd "" "${out}" 0
  out="${OUT_ROOT}/tum_${SEQ}_acc"
  mkdir -p "${out}"
  run_slam tum "${SEQ}" rgbdtsdf hybrid \
    "--enable-mapping --keyframe-tracking-interval 10 --feature-interval 5" "${out}" 0
done

SEQ=freiburg2_large_no_loop
if [ ! -d "${TUM_ROOT_FR2}/rgbd_dataset_${SEQ}" ]; then
  echo "==[${HOST}]== Extract fr2 .tgz → ${TUM_ROOT_FR2}"
  mkdir -p "${TUM_ROOT_FR2}"
  tar -xzf "${TUM_ROOT}/freiburg2_large_no_loop/raw_tgz/rgbd_dataset_${SEQ}.tgz" -C "${TUM_ROOT_FR2}"
fi
out="${OUT_ROOT}/tum_${SEQ}_speed"
mkdir -p "${out}"
run_slam_root() {
  local ds="$1" seq="$2" backend="$3" mode="$4" extra="$5" outdir="$6" cap="$7" root="$8"
  echo "==[${HOST}]== SLAM ${ds}/${seq}  root=${root}  backend=${backend}  mode=${mode}  cap=${cap:-0}"
  local cmd=(python scripts/run_slam.py "${ds}")
  cmd+=( --dataset-root "${root}" --sequence "${seq}" --slam-backend "${backend}" --tracking-mode "${mode}" --output "${outdir}" )
  if [ -n "${extra}" ]; then
    # shellcheck disable=SC2206
    extra_arr=( ${extra} )
    cmd+=( "${extra_arr[@]}" )
  fi
  if [ -n "${cap}" ] && [ "${cap}" -gt 0 ]; then
    cmd+=( --max-frames "${cap}" )
  fi
  "${cmd[@]}" 2>&1 | tee "${outdir}.log" | tail -25 || true
}
# The fr2 .tgz extracts to <root>/rgbd_dataset_<short>/ — pass the full
# name so the TUM loader finds it directly under basedir.
run_slam_root tum "rgbd_dataset_${SEQ}" fast_rgbd fast_rgbd "" "${out}" 1000 "${TUM_ROOT_FR2}"
out="${OUT_ROOT}/tum_${SEQ}_acc"
mkdir -p "${out}"
run_slam_root tum "rgbd_dataset_${SEQ}" rgbdtsdf hybrid \
  "--enable-mapping --keyframe-tracking-interval 10 --feature-interval 5" "${out}" 1000 "${TUM_ROOT_FR2}"

for CFG in speed acc; do
  out="${OUT_ROOT}/orbbec_${CFG}"
  mkdir -p "${out}"
  if [ "${CFG}" = "speed" ]; then
    run_slam normalized _ fast_rgbd fast_rgbd "" "${out}" 1500
  else
    run_slam normalized _ rgbdtsdf hybrid \
      "--enable-mapping --keyframe-tracking-interval 10 --feature-interval 5" "${out}" 1500
  fi
done

echo "==[${HOST}]== DONE. metrics.json files:"
find "${OUT_ROOT}" -name metrics.json -print
