#!/usr/bin/env bash

set -ev

if [[ -z $GITHUB_ACTION ]]; then
  ruff format mlenergy tests scripts
else
  ruff format --check mlenergy tests scripts
fi

ruff check mlenergy tests scripts
pyright mlenergy tests scripts
