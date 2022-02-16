#!/usr/bin/env bash
set -o errexit

start() { echo "::group::$1"; }
end() { echo "::endgroup::"; }
die() { set +v; echo "$*" 1>&2 ; sleep 1; exit 1; }


start changelog
echo 'TODO: Edit changelog with each PR?'
end changelog

start flake8
flake8 || die "Try: autopep8 --in-place --aggressive -r . --exclude $EXCLUDE"
end flake8

start pytest
PYTHONPATH=. coverage run --module pytest . -vv --doctest-modules
coverage report --show-missing --fail-under 90  # TODO: Ratchet this up.
end pytest
