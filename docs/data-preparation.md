# Data Preparation

Datasets not mentioned here are automatically downloaded and processed by the benchmark script.
If you want to control where they are stored, set the `HF_HOME` environment variable before running the benchmark.

The following datasets require manual download and preparation.

## Audio Dataset

We use the NVIDIA AudioSkills dataset's FSD50K subset for the `audio-chat` workload.
AudioSkills (for the text prompt) is downloaded automatically from Hugging Face Hub, but FSD50K (for the actual audio clips) needs to be downloaded separately from its official source.

FSD50K will take up about 33GB of disk space.

```bash
python scripts/prepare_fsd50k.py /data/fsd50k
export AUDIO_DATA_DIR=/data/fsd50k/FSD50K.dev_audio
```

## Video Dataset

We use the LLaVA-Video-178K dataset for the `video-chat` workload.
The whole dataset is about 1.2TB in size and requires about 2.4TB of free space during extraction.

The video's tarballs will be pulled from Hugging Face Hub, so set the `HF_HOME` environment variable appropriately if you want to control where they first hit the disk.
After that, tarballs will be extracted into the target directory you provide.

```bash
python scripts/prepare_llava_videos.py /data/llava-video-178k --jobs 12
export VIDEO_DATA_DIR=/data/llava-video-178k
```
