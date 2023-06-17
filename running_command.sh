#!/bin/bash

# node with four gpus
python benchmark.py --model-path /data/leaderboard/weights/lmsys/vicuna-7B --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json
python benchmark.py --model-path /data/leaderboard/weights/lmsys/vicuna-13B --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json --device-index 1
python benchmark.py --model-path /data/leaderboard/weights/tatsu-lab/alpaca-7B --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json --device-index 2
python benchmark.py --model-path /data/leaderboard/weights/metaai/llama-7B --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json --device-index 3

python benchmark.py --model-path /data/leaderboard/weights/metaai/llama-13B --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json
python benchmark.py --model-path camel-ai/CAMEL-13B-Combined-Data --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json --device-index 1
python benchmark.py --model-path /data/leaderboard/weights/BlinkDL/RWKV-4-Raven-7B-v12-Eng98%-Other2%-20230521-ctx8192.pth --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json --device-index 2
python benchmark.py --model-path databricks/dolly-v2-12b --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json --device-index 3

python benchmark.py --model-path FreedomIntelligence/phoenix-inst-chat-7b --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json
python benchmark.py --model-path h2oai/h2ogpt-gm-oasst1-en-2048-open-llama-7b-preview-300bt-v2 --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json --device-index 1
python benchmark.py --model-path lmsys/fastchat-t5-3b-v1.0 --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json --device-index 2
python benchmark.py --model-path Neutralzz/BiLLa-7B-SFT --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json --device-index 3

python benchmark.py --model-path nomic-ai/gpt4all-13b-snoozy --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json
python benchmark.py --model-path openaccess-ai-collective/manticore-13b-chat-pyg --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json --device-index 1
python benchmark.py --model-path OpenAssistant/oasst-sft-1-pythia-12b --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json --device-index 2
python benchmark.py --model-path project-baize/baize-v2-7B --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json --device-index 3

python benchmark.py --model-path /data/leaderboard/weights/BAIR/koala-7b --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json
python benchmark.py --model-path /data/leaderboard/weights/BAIR/koala-13b --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json --device-index 1
python benchmark.py --model-path StabilityAI/stablelm-tuned-alpha-7b --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json --device-index 2
python benchmark.py --model-path togethercomputer/RedPajama-INCITE-7B-Chat --input-file /data/leaderboard/sharegpt/sg_90k_part1_html_cleaned_lang_first_sampled.json --device-index 3
