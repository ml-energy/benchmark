#!/usr/bin/env bash

QUEUE_FILE="$1"

for model in $(tail -n +4 $QUEUE_FILE  | awk '{print $2}'); do
  HF_HOME=/data/leaderboard/hfcache huggingface-cli download $model --revision $(cat models/$model/revision.txt)
done
