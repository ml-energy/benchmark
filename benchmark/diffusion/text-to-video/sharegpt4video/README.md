# ShareGPT4Video dataset

For the text-to-video task, we sample 100 video captions from the ShareGPT4Video datset to feed to the diffusion model to generate videos.

## Filtering the dataset

Download the dataset with captions and video paths.

```sh
wget https://huggingface.co/datasets/ShareGPT4Video/ShareGPT4Video/resolve/main/sharegpt4video_40k.jsonl
```

Sample video-caption pairs. The sampled dataset will be saved under `sharegpt4video_100.json`.

```sh
python sample.py
```
