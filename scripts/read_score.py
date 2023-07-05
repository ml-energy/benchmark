import pandas as pd
import os
import csv

folder = "nlp"
folders = os.listdir(folder)
out_csv = csv.writer(open("score.csv", "w", newline=""))
for model in folders:
    tasks = os.listdir(folder+"/"+str(model))
    scores = []
    for task in tasks:
        df = pd.read_json(folder+"/"+str(model)+"/"+str(task))
        model_args = df['config']['model_args']
        results=df['results']
        keys = results.keys()
        if str(keys[0]) == "truthfulqa_mc":
            score = results=df['results'][keys[0]]['mc2']
        else:
            score = results=df['results'][keys[0]]['acc_norm']
        num_fewshot = df['config']['num_fewshot']
        scores.append(score)
    out_csv.writerow([model] + scores)
