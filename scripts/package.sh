#!/usr/bin/env bash
# kv3d package builder — produces a self-contained Linux tarball.
set -euo pipefail

VERSION="${KV3D_VERSION:-0.1.0}"
BUILD_DIR="${BUILD_DIR:-./build}"
DIST_DIR="${DIST_DIR:-./dist}"
OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)
PACKAGE_NAME="kv3d-${VERSION}-${OS}-${ARCH}"

echo "kv3d packager"
echo "  version  : $VERSION"
echo "  build_dir: $BUILD_DIR"
echo "  output   : $DIST_DIR/$PACKAGE_NAME.tar.gz"

# Ensure the binary exists
BINARY="$BUILD_DIR/kv3d"
if [ ! -f "$BINARY" ]; then
    BINARY="$BUILD_DIR/src/kv3d"
fi
if [ ! -f "$BINARY" ]; then
    echo "Error: kv3d binary not found. Run: cmake --build $BUILD_DIR -j"
    exit 1
fi

# Strip the binary for smaller distribution
strip "$BINARY" 2>/dev/null || true

mkdir -p "$DIST_DIR/$PACKAGE_NAME"

# Copy files
cp "$BINARY" "$DIST_DIR/$PACKAGE_NAME/kv3d"
cp scripts/install.sh "$DIST_DIR/$PACKAGE_NAME/"
mkdir -p "$DIST_DIR/$PACKAGE_NAME/docs"

# Create archive
(cd "$DIST_DIR" && tar -czf "${PACKAGE_NAME}.tar.gz" "$PACKAGE_NAME")
rm -rf "$DIST_DIR/$PACKAGE_NAME"

echo "Package created: $DIST_DIR/$PACKAGE_NAME.tar.gz"
ls -lh "$DIST_DIR/$PACKAGE_NAME.tar.gz"
