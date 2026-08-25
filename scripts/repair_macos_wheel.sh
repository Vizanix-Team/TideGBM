#!/usr/bin/env bash
# Bundle Homebrew libomp into the macOS wheel (no delocate — OpenMP breaks it on arm64).
set -euo pipefail

wheel="$1"
dest_dir="$2"
# $3 = {delocate_archs}, unused: we ship a single-arch wheel from native builds.

libomp_prefix="$(brew --prefix libomp)"
libomp_dylib="${libomp_prefix}/lib/libomp.dylib"
out_wheel="${dest_dir}/$(basename "${wheel}")"

if [[ ! -f "$libomp_dylib" ]]; then
  echo "libomp not found at ${libomp_dylib}" >&2
  exit 1
fi

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT
unzip -q "$wheel" -d "$tmp"

core="$(find "$tmp" -path '*/tidegbm/_core*.so' | head -1)"
if [[ -z "$core" ]]; then
  echo "tidegbm._core extension not found in ${wheel}" >&2
  exit 1
fi

echo "repair: $(file "$core")"

mkdir -p "${tmp}/tidegbm/.dylibs"
cp "$libomp_dylib" "${tmp}/tidegbm/.dylibs/libomp.dylib"
install_name_tool -id "@loader_path/libomp.dylib" "${tmp}/tidegbm/.dylibs/libomp.dylib"

while IFS= read -r old; do
  [[ -z "$old" ]] && continue
  [[ "$old" == "@loader_path/.dylibs/libomp.dylib" ]] && continue
  install_name_tool -change "$old" "@loader_path/.dylibs/libomp.dylib" "$core"
done < <(otool -L "$core" | awk '/libomp/{print $1}')

(
  cd "$tmp"
  zip -q -r "$out_wheel" tidegbm tidegbm-*.dist-info
)

echo "repaired wheel -> ${out_wheel}"
otool -L "$core" | head -8
