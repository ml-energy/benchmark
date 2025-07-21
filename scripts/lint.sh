#!/usr/bin/env bash

set -ev

if [[ -z $GITHUB_ACTION ]]; then
  ruff format mlenergy
else
  ruff format --check mlenergy
fi

ruff check mlenergy
pyright mlenergy
