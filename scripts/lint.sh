#!/usr/bin/env bash

set -ev

if [[ -z $GITHUB_ACTION ]]; then
  ruff format mlenergy tests
else
  ruff format --check mlenergy tests
fi

ruff check mlenergy tests
pyright mlenergy tests
