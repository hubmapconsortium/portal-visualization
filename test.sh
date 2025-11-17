#!/usr/bin/env bash
set -o errexit

start() { echo "::group::$1"; }
end() { echo "::endgroup::"; }
die() { set +v; echo "$*" 1>&2 ; sleep 1; exit 1; }


start changelog
echo 'TODO: Edit changelog with each PR?'
end changelog

start docs
diff README.md <(src/vis-preview.py --help) | grep '^>' && die "Update vis-preview.py docs in README.md"
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
