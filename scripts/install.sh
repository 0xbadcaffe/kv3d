#!/usr/bin/env bash
# kv3d install script
# Usage: curl -fsSL https://install.kv3d.dev | bash
set -euo pipefail

KV3D_VERSION="${KV3D_VERSION:-latest}"
INSTALL_DIR="${KV3D_INSTALL_DIR:-/usr/local/bin}"
REPO="https://github.com/your-org/kv3d-engine"

OS=$(uname -s | tr '[:upper:]' '[:lower:]')
ARCH=$(uname -m)

case "$ARCH" in
    x86_64)  ARCH_LABEL="x86_64" ;;
    aarch64) ARCH_LABEL="aarch64" ;;
    arm64)   ARCH_LABEL="aarch64" ;;
    *)       echo "Unsupported architecture: $ARCH"; exit 1 ;;
esac

case "$OS" in
    linux)  OS_LABEL="linux" ;;
    darwin) OS_LABEL="darwin" ;;
    *)      echo "Unsupported OS: $OS"; exit 1 ;;
esac

echo "kv3d installer"
echo "  version : $KV3D_VERSION"
echo "  platform: $OS_LABEL/$ARCH_LABEL"
echo "  target  : $INSTALL_DIR/kv3d"
echo ""

if [ "$KV3D_VERSION" = "latest" ]; then
    DOWNLOAD_URL="$REPO/releases/latest/download/kv3d-${OS_LABEL}-${ARCH_LABEL}.tar.gz"
else
    DOWNLOAD_URL="$REPO/releases/download/${KV3D_VERSION}/kv3d-${OS_LABEL}-${ARCH_LABEL}.tar.gz"
fi

TMP=$(mktemp -d)
trap "rm -rf $TMP" EXIT

echo "Downloading $DOWNLOAD_URL ..."
curl -fsSL "$DOWNLOAD_URL" -o "$TMP/kv3d.tar.gz"

echo "Extracting..."
tar -xzf "$TMP/kv3d.tar.gz" -C "$TMP"

echo "Installing to $INSTALL_DIR ..."
if [ -w "$INSTALL_DIR" ]; then
    cp "$TMP/kv3d" "$INSTALL_DIR/kv3d"
else
    sudo cp "$TMP/kv3d" "$INSTALL_DIR/kv3d"
fi
chmod +x "$INSTALL_DIR/kv3d"

echo ""
echo "kv3d installed successfully."
echo "Run 'kv3d doctor' to verify your setup."
