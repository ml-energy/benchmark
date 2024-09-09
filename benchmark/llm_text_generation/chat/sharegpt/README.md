# ShareGPT benchmarking dataset

## Download cleaned ShareGPT dataset

```sh
https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json
```

## Construct benchmarking dataset

Filter conversations with too long prompts/responses, conversations not started by "human", extract first turn, and randomly sample 500 prompts

```sh
pip install transformers
python filter_dataset.py
```
