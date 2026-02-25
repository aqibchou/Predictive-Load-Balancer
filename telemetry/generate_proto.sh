#!/usr/bin/env bash
# Regenerate Python gRPC bindings from telemetry.proto
# Run from the project root: bash telemetry/generate_proto.sh
#
# Prerequisites:
#   pip install grpcio-tools>=1.62.0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "Generating proto bindings from $SCRIPT_DIR/telemetry.proto ..."

python3 -m grpc_tools.protoc \
    --proto_path="$SCRIPT_DIR" \
    --python_out="$SCRIPT_DIR" \
    --grpc_python_out="$SCRIPT_DIR" \
    "$SCRIPT_DIR/telemetry.proto"

# grpc_tools.protoc emits a bare "import telemetry_pb2" which only works
# when the telemetry/ directory is on sys.path directly. Patch it to a
# package-relative import so it works when imported as "from telemetry import ...".
sed -i.bak \
    's/^import telemetry_pb2 as telemetry__pb2$/from telemetry import telemetry_pb2 as telemetry__pb2/' \
    "$SCRIPT_DIR/telemetry_pb2_grpc.py"
rm -f "$SCRIPT_DIR/telemetry_pb2_grpc.py.bak"

echo "Generated:"
echo "  $SCRIPT_DIR/telemetry_pb2.py"
echo "  $SCRIPT_DIR/telemetry_pb2_grpc.py"
echo "Done."
