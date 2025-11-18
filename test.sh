#!/usr/bin/env bash
set -o errexit

start() { echo "::group::$1"; }
end() { echo "::endgroup::"; }
die() { set +v; echo "$*" 1>&2 ; sleep 1; exit 1; }

start changelog
echo 'TODO: Edit changelog with each PR?'
end changelog

start docs
# Compare README with vis-preview --help output after installing the package
# Skip this check if running with thin install (no full dependencies)
if python -c "import sys; import vitessce" 2>/dev/null; then
    pip install -q -e . > /dev/null 2>&1 || true
    diff README.md <(vis-preview --help 2>&1 | grep -v "UserWarning" | grep -v "warn(" | grep -v "vitessce/__init__.py" || python -m portal_visualization.cli --help 2>&1 | grep -v "UserWarning" | grep -v "warn(" | grep -v "vitessce/__init__.py") | grep '^>' && die "Update vis-preview docs in README.md"
else
    echo "Skipping vis-preview docs check (requires [full] install)"
fi
end docs

start ruff-check
uv run ruff check src/ test/ || die "Run: ruff check --fix src/ test/"
end ruff-check

start ruff-format
uv run ruff format --check src/ test/ || die "Run: ruff format src/ test/"
end ruff-format

start pytest
PYTHONPATH=. uv run coverage run --module pytest . -vv --doctest-modules
uv run coverage report --show-missing --fail-under 100
end pytest
