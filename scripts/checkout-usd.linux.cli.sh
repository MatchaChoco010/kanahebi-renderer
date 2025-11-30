#! /usr/bin/env bash
set -euo pipefail

# settings
USD_REPO_DIR="external/OpenUSD"

# checkout USD
cd "$USD_REPO_DIR"
git checkout .
git checkout "v25.11"
cd -
