#!/usr/bin/env bash

set -ev

if [[ -z $GITHUB_ACTION ]]; then
  ruff format .
else
  ruff format --check .
fi

ruff check .
pyright .
