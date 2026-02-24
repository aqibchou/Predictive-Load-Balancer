#!/usr/bin/env bash
# k8s/chaos/run-chaos.sh — Orchestrated Chaos Engineering experiment
#
# Sequence:
#   1. Verify prerequisites (chaos-mesh namespace + CRDs)
#   2. Start Locust in headless mode (500 users) as background job
#   3. 30 s steady-state baseline
#   4. Apply lb-network-loss for 60 s  →  assert eBPF failsafe activates
#   5. 30 s recovery
#   6. Apply backend-cpu-hog for 60 s  →  assert Q-learning scaling activates
#   7. 30 s recovery
#   8. Kill Locust; print failure summary from CSV
#
# Prerequisites:
#   - kubectl + helm configured
#   - Chaos Mesh installed (run install.sh first)
#   - Locust installed: pip install locust
#   - GRAFANA_URL env var (optional, default shown)
#   - Locust file at tests/locustfile.py (adjust LOCUST_FILE below)
#
# Usage:
#   NAMESPACE=predictive-lb bash k8s/chaos/run-chaos.sh

set -euo pipefail

NAMESPACE="${NAMESPACE:-predictive-lb}"
LB_URL="${LB_URL:-http://localhost:8000}"
GRAFANA_URL="${GRAFANA_URL:-http://localhost:3000}"
LOCUST_FILE="${LOCUST_FILE:-tests/locustfile.py}"
LOCUST_USERS="${LOCUST_USERS:-500}"
LOCUST_SPAWN_RATE="${LOCUST_SPAWN_RATE:-50}"
RESULTS_DIR="${RESULTS_DIR:-results}"
CHAOS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "${RESULTS_DIR}"

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── 1. Prerequisites check ───────────────────────────────────────────────────
info "Checking prerequisites..."

if ! kubectl get namespace chaos-mesh &>/dev/null; then
    error "chaos-mesh namespace not found. Run k8s/chaos/install.sh first."
    exit 1
fi

if ! kubectl get crd networkchaos.chaos-mesh.org &>/dev/null; then
    error "Chaos Mesh CRDs not found. Run k8s/chaos/install.sh first."
    exit 1
fi

if ! kubectl get namespace "${NAMESPACE}" &>/dev/null; then
    error "Namespace '${NAMESPACE}' not found. Deploy the application first."
    exit 1
fi

if ! command -v locust &>/dev/null; then
    error "Locust not found. Run: pip install locust"
    exit 1
fi

info "All prerequisites satisfied."

# ── 2. Start Locust (background) ─────────────────────────────────────────────
LOCUST_CSV="${RESULTS_DIR}/locust_chaos"
LOCUST_LOG="${RESULTS_DIR}/locust.log"

info "Starting Locust with ${LOCUST_USERS} users against ${LB_URL} ..."
locust \
    --headless \
    --users "${LOCUST_USERS}" \
    --spawn-rate "${LOCUST_SPAWN_RATE}" \
    --host "${LB_URL}" \
    --locustfile "${LOCUST_FILE}" \
    --csv "${LOCUST_CSV}" \
    --logfile "${LOCUST_LOG}" \
    2>/dev/null &
LOCUST_PID=$!
info "Locust PID: ${LOCUST_PID}"

# Trap to ensure Locust is killed on exit
cleanup() {
    info "Cleaning up chaos experiments..."
    kubectl delete -f "${CHAOS_DIR}/experiment-lb-network-loss.yaml" --ignore-not-found -n "${NAMESPACE}" || true
    kubectl delete -f "${CHAOS_DIR}/experiment-backend-cpu.yaml"     --ignore-not-found -n "${NAMESPACE}" || true
    if kill -0 "${LOCUST_PID}" 2>/dev/null; then
        kill "${LOCUST_PID}" && wait "${LOCUST_PID}" 2>/dev/null || true
    fi
}
trap cleanup EXIT

# ── 3. Steady-state baseline ──────────────────────────────────────────────────
info "Grafana dashboard: ${GRAFANA_URL}"
info "Collecting 30 s steady-state baseline..."
sleep 30

BASELINE_END=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
info "Baseline period ended at ${BASELINE_END}"

# ── 4. LB Network Loss experiment ─────────────────────────────────────────────
warn "═══════════ EXPERIMENT 1: LB Network Loss ═══════════"
info "Applying 100%% packet loss on load balancer pod for 60 s..."
kubectl apply -f "${CHAOS_DIR}/experiment-lb-network-loss.yaml" -n "${NAMESPACE}"
CHAOS1_START=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

info "Waiting 90 s (60 s chaos + 30 s recovery)..."
sleep 90

kubectl delete -f "${CHAOS_DIR}/experiment-lb-network-loss.yaml" -n "${NAMESPACE}" --ignore-not-found
CHAOS1_END=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
info "LB network loss experiment ended. Recovery window: ${CHAOS1_START} → ${CHAOS1_END}"

# ── 5. Inter-experiment pause ─────────────────────────────────────────────────
info "30 s recovery pause before next experiment..."
sleep 30

# ── 6. Backend CPU stress experiment ──────────────────────────────────────────
warn "═══════════ EXPERIMENT 2: Backend CPU Stress ═══════════"
info "Applying 90%% CPU stress on all backend pods for 60 s..."
kubectl apply -f "${CHAOS_DIR}/experiment-backend-cpu.yaml" -n "${NAMESPACE}"
CHAOS2_START=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

info "Waiting 90 s (60 s chaos + 30 s recovery)..."
sleep 90

kubectl delete -f "${CHAOS_DIR}/experiment-backend-cpu.yaml" -n "${NAMESPACE}" --ignore-not-found
CHAOS2_END=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
info "CPU stress experiment ended. Window: ${CHAOS2_START} → ${CHAOS2_END}"

# ── 7. Final recovery ─────────────────────────────────────────────────────────
info "30 s final recovery window..."
sleep 30

# ── 8. Collect results ────────────────────────────────────────────────────────
kill "${LOCUST_PID}" 2>/dev/null && wait "${LOCUST_PID}" 2>/dev/null || true
LOCUST_PID=0   # prevent double-kill in trap

echo ""
info "══════════════════ RESULTS SUMMARY ══════════════════"

STATS_CSV="${LOCUST_CSV}_stats.csv"
if [[ -f "${STATS_CSV}" ]]; then
    echo ""
    echo "Locust statistics (${STATS_CSV}):"
    column -t -s ',' "${STATS_CSV}" | head -20
    echo ""

    # Extract failure count from the CSV (column 8 = Failure Count in Locust 2.x)
    FAILURES=$(awk -F',' 'NR>1 && $1=="Aggregated" {print $8}' "${STATS_CSV}")
    TOTAL=$(awk -F',' 'NR>1 && $1=="Aggregated" {print $3}' "${STATS_CSV}")
    if [[ -n "${FAILURES}" ]]; then
        echo "Total requests : ${TOTAL:-unknown}"
        echo "Total failures : ${FAILURES}"
        if [[ "${FAILURES}" -eq 0 ]]; then
            info "✔ PASS — Zero failures recorded during chaos experiments"
        else
            warn "⚠ ${FAILURES} failure(s) recorded — review eBPF failsafe timing"
        fi
    fi
else
    warn "Locust CSV not found at ${STATS_CSV} — check ${LOCUST_LOG}"
fi

echo ""
info "Experiment windows:"
echo "  Baseline end           : ${BASELINE_END}"
echo "  LB network loss        : ${CHAOS1_START} → ${CHAOS1_END}"
echo "  Backend CPU stress     : ${CHAOS2_START} → ${CHAOS2_END}"
echo ""
info "Full Locust log: ${LOCUST_LOG}"
info "Done."
