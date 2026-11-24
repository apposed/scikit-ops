#!/bin/bash
# Release scikit-ops and its helper packages to PyPI.
#
#   scripts/release.sh                       # release any package whose version is not on PyPI yet
#   scripts/release.sh oprunner scikit-ops   # bump their .devN first, commit, then release
#
# Helpers go up before scikit-ops, which depends on them. Needs uv and a PyPI
# token (uv publish asks, or reads UV_PUBLISH_TOKEN).
set -euo pipefail
cd "$(dirname "$0")/.."

# package name -> folder, in release order
names=(opspec ophelper oprunner scikit-ops)
declare -A dirs=([opspec]=opspec [ophelper]=ophelper [oprunner]=oprunner [scikit-ops]=.)

version_of() { sed -n 's/^version = "\(.*\)"$/\1/p' "${dirs[$1]}/pyproject.toml" | head -1; }
on_pypi() { curl -sf -o /dev/null "https://pypi.org/pypi/$1/$2/json"; }

git pull --ff-only
if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
    echo "Uncommitted changes. Commit or discard them first." >&2
    exit 1
fi

# Bump the named packages: 0.1.0.devN -> 0.1.0.dev(N+1)
if [ $# -gt 0 ]; then
    bumped=()
    for name in "$@"; do
        [ -n "${dirs[$name]:-}" ] || { echo "Unknown package: $name" >&2; exit 1; }
        old=$(version_of "$name")
        [[ $old =~ ^(.*\.dev)([0-9]+)$ ]] || { echo "Can't bump $name $old" >&2; exit 1; }
        new="${BASH_REMATCH[1]}$((BASH_REMATCH[2] + 1))"
        sed -i "0,/^version = \"$old\"/s//version = \"$new\"/" "${dirs[$name]}/pyproject.toml"
        bumped+=("$name $new")
    done
    uv lock
    git commit -q -am "Release $(IFS=,; echo "${bumped[*]}" | sed 's/,/, /g')"
fi

todo=()
for name in "${names[@]}"; do
    v=$(version_of "$name")
    on_pypi "$name" "$v" || todo+=("$name")
done
if [ ${#todo[@]} -eq 0 ]; then
    echo "Every version is already on PyPI. Name the packages to bump, e.g. scripts/release.sh oprunner"
    exit 0
fi

echo "Release from $(git log --oneline -1):"
for name in "${todo[@]}"; do echo "  $name $(version_of "$name")"; done
read -rp "Push main, tag and upload? [y/N] " ok
[ "$ok" = y ] || exit 1

git push origin main
for name in "${todo[@]}"; do
    v=$(version_of "$name")
    dir=${dirs[$name]}
    git tag "$name-v$v"
    rm -rf "$dir/dist"
    (cd "$dir" && uv build --out-dir dist)
    uv publish --check-url https://pypi.org/simple/ "$dir"/dist/*
done
git push origin --tags

echo "Done. Released: ${todo[*]}"
