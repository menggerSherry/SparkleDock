#!/bin/bash
set -euo pipefail
set -x

cd "$(dirname "$0")/../"
dock_home="$(pwd)"

COMPLEXES=("2VXT" "3VLB" "2A1A" "2GTP" "2X9A" "1RKE" "4GAM" "4JCV" "4LW4")

BM_HOME="${dock_home}/bm"
STEPS=100
WARMUP=1
RUNS=5
NP=1

export LD_LIBRARY_PATH="${dock_home}/lib:${LD_LIBRARY_PATH:-}"
export NUM_THREADS=32
export USE_CUDA=1

RAW_CSV="${BM_HOME}/gpu_measure_raw.csv"
SUMMARY_CSV="${BM_HOME}/gpu_speedup_summary.csv"
LOG_DIR="${BM_HOME}/gpu_logs"

mkdir -p "${LOG_DIR}"

echo "complex,phase,rep,np,steps,gpu_time_sec,status,log_file" > "${RAW_CSV}"

run_once() {
    local complex="$1"
    local phase="$2"
    local rep="$3"

    local workdir="${BM_HOME}/${complex}"
    local log_file="${LOG_DIR}/${complex}_${phase}_${rep}.log"

    cd "${workdir}"

    rm -rf lightdock* init/ swarm_* clustered

    local start_ts
    local end_ts
    local elapsed
    local status

    start_ts="$(date +%s.%N)"

    set +e
    mpirun -np "${NP}" "${dock_home}/bin/sparkledock" -f setup.json -s "${STEPS}" -l 1 > "${log_file}" 2>&1
    status=$?
    set -e

    end_ts="$(date +%s.%N)"

    elapsed="$(python3 - <<PY
start=${start_ts}
end=${end_ts}
print(f"{end-start:.6f}")
PY
)"

    echo "${complex},${phase},${rep},${NP},${STEPS},${elapsed},${status},${log_file}" >> "${RAW_CSV}"

    if [ "${status}" -ne 0 ]; then
        echo "ERROR: ${complex} ${phase} ${rep} failed. See ${log_file}"
        return "${status}"
    fi
}

for complex in "${COMPLEXES[@]}"; do
    for rep in $(seq 1 "${WARMUP}"); do
        run_once "${complex}" "warmup" "${rep}"
    done

    for rep in $(seq 1 "${RUNS}"); do
        run_once "${complex}" "measure" "${rep}"
    done
done

python3 - <<PY
import csv
import math
from collections import defaultdict

raw_csv = "${RAW_CSV}"
summary_csv = "${SUMMARY_CSV}"

cpu_time = {
    "2VXT": 1420.0,
    "3VLB": 592.0,
    "2A1A": 328.0,
    "2GTP": 304.0,
    "2X9A": 30.0,
    "1RKE": 285.0,
    "4GAM": 7568.0,
    "4JCV": 2391.0,
    "4LW4": 1284.0,
}

t95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
}

groups = defaultdict(list)

with open(raw_csv, newline="") as f:
    reader = csv.DictReader(f)
    for row in reader:
        if row["phase"] != "measure":
            continue
        if row["status"] != "0":
            continue
        groups[row["complex"]].append(float(row["gpu_time_sec"]))

with open(summary_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "complex",
        "cpu_time_sec_fixed",
        "n_gpu_runs",
        "gpu_mean_sec",
        "gpu_std_sec",
        "gpu_ci95_half_sec",
        "speedup_mean",
        "speedup_std",
        "speedup_ci95_half",
        "speedup_ci95_low",
        "speedup_ci95_high",
    ])

    avg_speedups = []

    for complex_name in cpu_time:
        values = groups.get(complex_name, [])
        if not values:
            continue

        n = len(values)
        gpu_mean = sum(values) / n

        if n > 1:
            gpu_var = sum((x - gpu_mean) ** 2 for x in values) / (n - 1)
            gpu_std = math.sqrt(gpu_var)
            t = t95.get(n - 1, 1.96)
            gpu_ci = t * gpu_std / math.sqrt(n)
        else:
            gpu_std = 0.0
            gpu_ci = 0.0
            t = 0.0

        speedups = [cpu_time[complex_name] / x for x in values]
        speedup_mean = sum(speedups) / n

        if n > 1:
            speedup_var = sum((x - speedup_mean) ** 2 for x in speedups) / (n - 1)
            speedup_std = math.sqrt(speedup_var)
            speedup_ci = t95.get(n - 1, 1.96) * speedup_std / math.sqrt(n)
        else:
            speedup_std = 0.0
            speedup_ci = 0.0

        avg_speedups.append(speedup_mean)

        writer.writerow([
            complex_name,
            f"{cpu_time[complex_name]:.6f}",
            n,
            f"{gpu_mean:.6f}",
            f"{gpu_std:.6f}",
            f"{gpu_ci:.6f}",
            f"{speedup_mean:.6f}",
            f"{speedup_std:.6f}",
            f"{speedup_ci:.6f}",
            f"{speedup_mean - speedup_ci:.6f}",
            f"{speedup_mean + speedup_ci:.6f}",
        ])

    if avg_speedups:
        avg = sum(avg_speedups) / len(avg_speedups)
        writer.writerow([])
        writer.writerow(["average_speedup_across_complexes", "", len(avg_speedups), "", "", "", f"{avg:.6f}", "", "", "", ""])

print(f"Wrote raw GPU results to {raw_csv}")
print(f"Wrote speedup summary to {summary_csv}")
PY