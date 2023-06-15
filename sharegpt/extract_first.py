import argparse
import json


def extract_first_sen(content):
    result = []
    for item in content:
        tmp = item
        tmp['conversations'] = [item['conversations'][0]]
        result.append(tmp)
    return result


def main(args):
    content = json.load(open(args["in_file"], "r"))
    content = extract_first_sen(content )
    json.dump(content, open(args["out_file"], "w"), indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in-file", type=str, default = 'sg_90k_part1_html_cleaned_lang.json' )
    parser.add_argument("--out-file", type=str, default = "sg_90k_part1_html_cleaned_lang_first.json")
    args = parser.parse_args()
    main(vars(args))


