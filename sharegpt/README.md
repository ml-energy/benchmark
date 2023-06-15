
## Download ShareGPT :
```
https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part1_html_cleaned.json

https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/HTML_cleaned_raw_dataset/sg_90k_part2_html_cleaned.json
```

## Install Fastchat
```
pip3 install fastchat
```

## Clean data:
```
pip3 install polyglot pyicu pycld2
python3 -m fastchat.data.optional_clean --in sg_90k_part1_html_cleaned.json --out sg_90k_part1_html_cleaned_lang.json --keep-lang en
```

## Extract first sentence (optional)
```
python extract_first.py --in-file sg_90k_part1_html_cleaned_lang.json --out-file sg_90k_part1_html_cleaned_lang_first.json
```

## Sample data (optional)
```
python3 -m fastchat.data.sample --in sg_90k_part1_html_cleaned_lang_first.json --out sg_90k_part1_html_cleaned_lang_first_sampled.json --end 10000 --max-length 10000
```

## ShareGPT Feeder Usage

```
from sharegpt_feeder import generator
sharegpt_generator = generator()
print(next(sharegpt_generator))
print(next(sharegpt_generator))
```