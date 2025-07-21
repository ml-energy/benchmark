#!/usr/bin/env bash

set -ev

if [[ -z $GITHUB_ACTION ]]; then
  ruff format benchmark
else
  ruff format --check benchmark
fi

ruff check benchmark
pyright benchmark
