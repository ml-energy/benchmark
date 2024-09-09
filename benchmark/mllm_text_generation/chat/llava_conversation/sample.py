import json
import random
import base64

SEED = 68
NUM_SAMPLES = 500


def main() -> None:
    random.seed(SEED)

    with open("full.json") as f:
        data = json.load(f)
        data = random.sample(data, NUM_SAMPLES * 2)

    dataset = []
    data_iter = iter(data)
    while len(dataset) < NUM_SAMPLES:
        sample = next(data_iter)

        # 1. The image should exist.
        # 2. Even index messages in the conversation should be from the human.
        # 3. The first message should contain at most one "<image>" substring, which will be removed.
        # 4. Even index messages will be concatenated to form the prompt.
        image_path = "train2017/" + sample["image"]
        conversation = []
        for conv in sample["conversations"][::2]:
            assert conv["from"] == "human", sample
            conversation.append(conv["value"])
        if (ind := conversation[0].find("<image>")) != -1:
            conversation[0] = conversation[0][:ind] + conversation[0][ind+len("<image>"):]
        
        message = ""
        for conv in conversation:
            assert "<image>" not in conv, sample
            message += conv.strip() + " "
        message = message.strip()

        dataset.append(
            dict(
                image=base64.b64encode(open(image_path, "rb").read()).decode("utf-8"),
                prompt=message,
            ),
        )

    with open(f"llava_conversation_{NUM_SAMPLES}.json", "w") as f:
        json.dump(dataset, f)


if __name__ == "__main__":
    main()
