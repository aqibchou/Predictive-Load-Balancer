#!/usr/bin/env bash
# k8s/chaos/install.sh — Install Chaos Mesh on a Kubernetes cluster
#
# Prerequisites:
#   - kubectl configured to target the right cluster
#   - helm 3.x installed
#
# Usage:
#   bash k8s/chaos/install.sh

set -euo pipefail

CHAOS_MESH_VERSION="${CHAOS_MESH_VERSION:-2.6.3}"
NAMESPACE="chaos-mesh"

echo "═══════════════════════════════════════════════════"
echo " Installing Chaos Mesh ${CHAOS_MESH_VERSION}"
echo "═══════════════════════════════════════════════════"

# ── 1. Add the Chaos Mesh Helm repo ─────────────────────────────────────────
helm repo add chaos-mesh https://charts.chaos-mesh.org
helm repo update

# ── 2. Create the chaos-mesh namespace ──────────────────────────────────────
kubectl create namespace "${NAMESPACE}" --dry-run=client -o yaml | kubectl apply -f -

# ── 3. Install Chaos Mesh ────────────────────────────────────────────────────
helm upgrade --install chaos-mesh chaos-mesh/chaos-mesh \
    --namespace "${NAMESPACE}" \
    --version "${CHAOS_MESH_VERSION}" \
    --set dashboard.create=true \
    --set chaosDaemon.runtime=containerd \
    --set chaosDaemon.socketPath=/run/containerd/containerd.sock \
    --wait --timeout 5m

echo ""
echo "✔ Chaos Mesh installed."
echo ""
echo "Dashboard (port-forward):"
echo "  kubectl port-forward -n ${NAMESPACE} svc/chaos-dashboard 2333:2333"
echo "  open http://localhost:2333"
echo ""
echo "Verify CRDs:"
echo "  kubectl get crd | grep chaos-mesh"
