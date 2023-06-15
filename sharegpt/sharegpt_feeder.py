''' Usage
sharegpt_generator = sharegpt_generator()
print(next(sharegpt_generator))
print(next(sharegpt_generator))
print(next(sharegpt_generator))
'''
import json

def sharegpt_generator(file = 'sg_90k_part1_html_cleaned_lang.json'):
    content = json.load(open(file, "r"))
    for item in content:
        yield item['conversations'][0]['value']



