import json
import random

DATASET_PATH = "sharegpt4video_40k.jsonl"
VIDEO_SHARD_NAME = "panda_videos_1.zip"
NUM_SAMPLES = 700
SEED = 1


def main() -> None:
    dataset = [json.loads(line) for line in open(DATASET_PATH) if VIDEO_SHARD_NAME in line]
    random.seed(SEED)
    random.shuffle(dataset)

    sampled = dict(caption=[], video_id=[])
    for sample in dataset[:NUM_SAMPLES]:
        assert sample["zip_folder"] == VIDEO_SHARD_NAME, f"sample from wrong video shard: {sample}"
        whole_video_caption = next(
            (c for c in sample["captions"] if c["idx"] == "-1"), None
        )
        assert whole_video_caption is not None, f"whole video caption not found for sample: {sample}"
        sampled["caption"].append(whole_video_caption["content"])
        sampled["video_id"].append(sample["video_id"])

    json.dump(sampled, open(f"sharegpt4video_{NUM_SAMPLES}.json", "w"))


if __name__ == "__main__":
    main()
