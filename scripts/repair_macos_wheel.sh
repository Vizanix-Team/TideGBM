#!/usr/bin/env bash
# Bundle Homebrew libomp into the wheel, then run delocate-wheel.
set -euo pipefail

wheel="$1"
dest_dir="$2"
archs="$3"

libomp_prefix="$(brew --prefix libomp)"
libomp_lib="${libomp_prefix}/lib"
libomp_dylib="${libomp_lib}/libomp.dylib"

if [[ ! -f "$libomp_dylib" ]]; then
  echo "libomp not found at ${libomp_dylib}" >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
unzip -q "$wheel" -d "$tmp"

core="$(find "$tmp" -path '*/tidegbm/_core*.so' | head -1)"
if [[ -z "$core" ]]; then
  echo "tidegbm._core extension not found in wheel" >&2
  exit 1
fi

mkdir -p "${tmp}/tidegbm/.dylibs"
cp "$libomp_dylib" "${tmp}/tidegbm/.dylibs/"

while IFS= read -r old; do
  [[ -z "$old" ]] && continue
  install_name_tool -change "$old" "@loader_path/.dylibs/libomp.dylib" "$core"
done < <(otool -L "$core" | awk '/libomp/{print $1}')

patched="$(mktemp "${TMPDIR:-/tmp}/tidegbm-wheel.XXXXXX.whl")"
(
  cd "$tmp"
  zip -q -r "$patched" tidegbm "tidegbm-"*.dist-info
)

export REPAIR_LIBRARY_PATH="${libomp_lib}"
DYLD_LIBRARY_PATH="${REPAIR_LIBRARY_PATH}" delocate-wheel \
  --require-archs "$archs" \
  -w "$dest_dir" \
  -v "$patched" \
  -L "$libomp_lib"
