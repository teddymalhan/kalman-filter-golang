#!/usr/bin/env bash
# build.sh — compile the Kalman Filter WASM demo and copy support files.
#
# Usage:
#   ./build.sh          # build only
#   ./build.sh --serve  # build then start a local server on :8080
set -euo pipefail

GOROOT=$(go env GOROOT)
OUT_DIR="web"

echo "▶ Building WASM..."
mkdir -p "$OUT_DIR"
GOARCH=wasm GOOS=js go build -o "$OUT_DIR/kalman.wasm" ./wasm/

echo "▶ Copying wasm_exec.js..."
# The file moved from misc/wasm/ to lib/wasm/ in Go 1.24.
if   [ -f "$GOROOT/lib/wasm/wasm_exec.js" ];  then
    cp "$GOROOT/lib/wasm/wasm_exec.js"  "$OUT_DIR/"
elif [ -f "$GOROOT/misc/wasm/wasm_exec.js" ]; then
    cp "$GOROOT/misc/wasm/wasm_exec.js" "$OUT_DIR/"
else
    echo "ERROR: cannot locate wasm_exec.js under GOROOT=$GOROOT"
    exit 1
fi

echo "✓ Build complete → $OUT_DIR/"
ls -lh "$OUT_DIR/"

if [ "${1:-}" = "--serve" ]; then
    echo ""
    echo "▶ Serving on http://localhost:8080 ..."
    cd "$OUT_DIR"
    python3 -m http.server 8080
fi
