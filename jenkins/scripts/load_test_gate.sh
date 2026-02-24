#!/usr/bin/env bash
# load_test_gate.sh — Jenkins PR failsafe load test gate
#
# Usage:
#   bash jenkins/scripts/load_test_gate.sh [LB_HOST]
#
# Starts the stack with docker-compose (isolated network), runs Locust in
# headless mode (500 users, 60 s), parses the CSV output, and asserts:
#   - P95 response time < 500 ms
#   - Error rate          < 1%
#
# Exit codes:
#   0 — gate passed
#   1 — gate failed (assertions not met or Locust did not produce output)
#
# Cleanup (docker-compose down) is intentionally left to the Jenkins post block
# so that logs are preserved on failure.

set -euo pipefail

LB_HOST="${1:-http://localhost:8000}"
USERS=500
SPAWN_RATE=50
RUN_TIME=60
PREFIX="results/locust_gate"
LOCUST_FILE="tests/locust/locustfile.py"

P95_THRESHOLD_MS=500
ERROR_RATE_THRESHOLD=0.01   # 1%

# ── Ensure results directory exists ───────────────────────────────────────────
mkdir -p results

echo "=== Load Test Gate ==="
echo "  Host        : ${LB_HOST}"
echo "  Users       : ${USERS}"
echo "  Spawn rate  : ${SPAWN_RATE}/s"
echo "  Duration    : ${RUN_TIME}s"
echo "  P95 limit   : ${P95_THRESHOLD_MS} ms"
echo "  Error limit : $(echo "${ERROR_RATE_THRESHOLD} * 100" | bc -l | xargs printf '%.0f')%"
echo ""

# ── Wait for load balancer to be healthy ──────────────────────────────────────
echo "[1/3] Waiting for load balancer to become healthy …"
MAX_WAIT=120
ELAPSED=0
until curl --silent --fail --max-time 5 "${LB_HOST}/health" > /dev/null 2>&1; do
    if [ "${ELAPSED}" -ge "${MAX_WAIT}" ]; then
        echo "ERROR: Load balancer did not become healthy within ${MAX_WAIT}s"
        exit 1
    fi
    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  … still waiting (${ELAPSED}s elapsed)"
done
echo "  Load balancer is healthy."

# ── Run Locust headless ────────────────────────────────────────────────────────
echo ""
echo "[2/3] Running Locust (${USERS} users, ${RUN_TIME}s) …"
locust \
    --locustfile  "${LOCUST_FILE}" \
    --host        "${LB_HOST}" \
    --headless \
    --users       "${USERS}" \
    --spawn-rate  "${SPAWN_RATE}" \
    --run-time    "${RUN_TIME}s" \
    --csv         "${PREFIX}" \
    --html        "${PREFIX}_report.html" \
    --only-summary

echo "  Locust run complete."

# ── Parse results ─────────────────────────────────────────────────────────────
echo ""
echo "[3/3] Parsing results …"

STATS_CSV="${PREFIX}_stats.csv"

if [ ! -f "${STATS_CSV}" ]; then
    echo "ERROR: Locust stats CSV not found at ${STATS_CSV}"
    exit 1
fi

echo "  Stats CSV: ${STATS_CSV}"
echo ""
echo "--- Raw CSV (header + Aggregated row) ---"
head -1 "${STATS_CSV}"
grep "Aggregated" "${STATS_CSV}" || true
echo "-----------------------------------------"
echo ""

# Locust --csv produces columns (1-indexed):
# 1:Type  2:Name  3:Request Count  4:Failure Count  5:Median
# 6:Average  7:Min  8:Max  9:Avg Content Size  10:Requests/s
# 11:Failures/s  12:50%  13:66%  14:75%  15:80%  16:90%  17:95%  ...
# The aggregated row has Type="" and Name="Aggregated"
P95_MS=$(awk -F',' 'NR>1 && $2=="Aggregated" {printf "%d", $17}' "${STATS_CSV}")
FAIL_COUNT=$(awk -F',' 'NR>1 && $2=="Aggregated" {print $4+0}' "${STATS_CSV}")
REQ_COUNT=$(awk  -F',' 'NR>1 && $2=="Aggregated" {print $3+0}' "${STATS_CSV}")

if [ -z "${P95_MS}" ] || [ -z "${REQ_COUNT}" ] || [ "${REQ_COUNT}" -eq 0 ]; then
    echo "ERROR: Could not parse Aggregated row from ${STATS_CSV} (REQ_COUNT=${REQ_COUNT:-0})"
    exit 1
fi

# Compute error rate flag using awk (bc alternative — avoids bc dependency)
ERROR_RATE_OVER=$(awk "BEGIN {
    rate = ${FAIL_COUNT} / ${REQ_COUNT};
    print (rate > ${ERROR_RATE_THRESHOLD}) ? 1 : 0
}")

echo "  Total requests : ${REQ_COUNT}"
echo "  Failures       : ${FAIL_COUNT}"
echo "  P95 latency    : ${P95_MS} ms  (limit: ${P95_THRESHOLD_MS} ms)"
echo "  Error rate flag: ${ERROR_RATE_OVER}  (1 = over threshold)"
echo ""

# ── Assertions ────────────────────────────────────────────────────────────────
GATE_FAILED=0

if [ "${P95_MS}" -gt "${P95_THRESHOLD_MS}" ]; then
    echo "FAIL: P95 latency ${P95_MS} ms exceeds threshold ${P95_THRESHOLD_MS} ms"
    GATE_FAILED=1
else
    echo "PASS: P95 latency ${P95_MS} ms <= ${P95_THRESHOLD_MS} ms"
fi

if [ "${ERROR_RATE_OVER}" -eq 1 ]; then
    ACTUAL_PCT=$(awk "BEGIN {printf \"%.2f\", ${FAIL_COUNT} / ${REQ_COUNT} * 100}")
    echo "FAIL: Error rate ${ACTUAL_PCT}% exceeds threshold $(echo "${ERROR_RATE_THRESHOLD} * 100" | awk '{printf "%.0f", $1}')%"
    GATE_FAILED=1
else
    ACTUAL_PCT=$(awk "BEGIN {printf \"%.2f\", ${FAIL_COUNT} / ${REQ_COUNT} * 100}")
    echo "PASS: Error rate ${ACTUAL_PCT}% <= 1%"
fi

echo ""
if [ "${GATE_FAILED}" -eq 1 ]; then
    echo "=== Load Test Gate: FAILED ==="
    exit 1
fi

echo "=== Load Test Gate: PASSED ==="
exit 0
