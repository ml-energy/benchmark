import re
import json
import numpy as np
import statistics
import os
import csv

model = []
throughput = []
response_length = []
latency = []
energy = []

temp_throughput = []
temp_response_length = []
temp_latency = []
temp_energy = []

model_name = os.listdir("data/chat")

match_name = False

for models in model_name:
    with open("data/chat/"+models+"/benchmark.json", 'r') as file:
        json_data = json.load(file)

    for obj in json_data:
        if not match_name:
            name = str(obj["model"])
            model.append(name.replace('--','/'))
            match_name = True
        temp_throughput.append(float(obj["throughput"]))
        temp_response_length.append(float(obj["response_length"]))
        temp_latency.append(float(obj["latency"]))
        temp_energy.append(float(obj["energy"]))
        
    match_name = False

    throughput.append(temp_throughput.copy())
    response_length.append(temp_response_length.copy())
    latency.append(temp_latency.copy())
    energy.append(temp_energy.copy())

    temp_throughput.clear()
    temp_response_length.clear()
    temp_latency.clear()
    temp_energy.clear()


avg_throughput = [statistics.mean(row) for row in throughput]
avg_response_length = [statistics.mean(row) for row in response_length]
avg_latency = [statistics.mean(row) for row in latency]
avg_energy = [statistics.mean(row) for row in energy]

for i in range(len(model)):
    print(model[i])
    print(len(throughput[i]))
    print(len(response_length[i]))
    print(len(latency[i]))
    print(len(energy[i]))

csv_file = "leaderboard.csv"

with open(csv_file, "w", newline="") as file:
    writer = csv.writer(file)
    writer.writerow(["model","throughput","response_length","latency","energy"])  
    for i in range(len(model)):
        writer.writerow([model[i], avg_throughput[i], avg_response_length[i], avg_latency[i], avg_energy[i]])

