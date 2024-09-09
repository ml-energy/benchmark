# Benchmarking dataset

500 prompt/image pairs were sampled from the `conversation` subset of the Llava-Instruct dataset.

## Obtaining and filtering the dataset

First, download the full conversatio dataset with 58k samples:

```sh
curl -L https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/resolve/9d451dc7629cfe0469f6ae4432b765cd603d5fcb/conversation_58k.json -o full.json
```

Also fetch all COCO trainset and extract images:

```sh
curl -LO http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip
```

Finally, run the script to (1) sample 500 pairs, (2) read in the corresponding COCO image, (3) encode images into base64 strings, and (4) merge all pairs into one JSON file.
After this step, original dataset files (Llava and COCO) may be deleted.

```sh
python sample.py
```
