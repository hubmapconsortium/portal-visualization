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
pip install -q -e . > /dev/null 2>&1 || true
diff README.md <(vis-preview --help 2>&1 | grep -v "UserWarning" | grep -v "warn(" | grep -v "vitessce/__init__.py" || python -m portal_visualization.cli --help 2>&1 | grep -v "UserWarning" | grep -v "warn(" | grep -v "vitessce/__init__.py") | grep '^>' && die "Update vis-preview docs in README.md"
end docs

start ruff-check
ruff check src/ test/ || die "Run: ruff check --fix src/ test/"
end ruff-check

start ruff-format
ruff format --check src/ test/ || die "Run: ruff format src/ test/"
end ruff-format

start pytest
PYTHONPATH=. coverage run --module pytest . -vv --doctest-modules
coverage report --show-missing --fail-under 100
end pytest
