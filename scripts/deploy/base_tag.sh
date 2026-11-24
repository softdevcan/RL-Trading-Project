#!/usr/bin/env bash
# Bagimlilik temel imajinin etiketi: requirements.txt + docker/base.Dockerfile icerik ozeti (ilk 12 hane).
# Ikisinden biri degismedikce etiket ayni kalir; CI/Deploy bu etiketteki temel imaj GHCR'de varsa yeniden derlemez.
# CR karakterleri atilir: Windows'ta (CRLF) ve Linux'ta (LF) ayni etiket cikar.
set -euo pipefail
cd "$(dirname "$0")/../.."
cat requirements.txt docker/base.Dockerfile | tr -d '\r' | sha256sum | cut -c1-12
