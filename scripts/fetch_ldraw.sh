#!/usr/bin/env bash
# Downloads the official LDraw parts library into data/ldraw (parts, primitives, colours).
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p data
curl -L --fail -o data/complete.zip https://library.ldraw.org/library/updates/complete.zip
(cd data && unzip -oq complete.zip 'ldraw/LDConfig.ldr' 'ldraw/parts/*' 'ldraw/p/*' && rm complete.zip)
echo "LDraw library installed in data/ldraw"
