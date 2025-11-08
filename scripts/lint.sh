#!/usr/bin/env bash

set -ev

if [[ -z $GITHUB_ACTION ]]; then
  ruff format mlenergy tests scripts/generate_jobs.py
else
  ruff format --check mlenergy tests scripts/generate_jobs.py
fi

ruff check mlenergy tests scripts/generate_jobs.py
pyright mlenergy tests scripts/generate_jobs.py
