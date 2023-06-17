import subprocess
import re
import matplotlib.pyplot as plt
import datetime
import numpy as np
import statistics
import pdb
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

model1 = input("model 1: ")
model2 = input("model 2: ")
model3 = input("model 3: ")
model4 = input("model 4: ")

model_name = []
model_name.append(model1)
model_name.append(model2)
model_name.append(model3)
model_name.append(model4)

match_name = False

for i in range(len(model_name)):
    with open(model_name[i], 'r') as file:
        model_lines = file.readlines()

    for i in range(len(model_lines)):
        match = re.search(r'"model":\s*"([^"]+)"', model_lines[i])
        match1 = re.search(r'"throughput":\s*(\d+.\d+)', model_lines[i])
        match2 = re.search(r'"response_length":\s*([0-9]+)', model_lines[i])
        match3 = re.search(r'"latency":\s*(\d+.\d+)', model_lines[i])
        match4 = re.search(r'"energy":\s*(\d+.\d+)', model_lines[i])
        if match and not match_name:
            temp_model_name = str(match.group(1))
            model.append(temp_model_name.replace('--', '/'))
            match_name = True
        elif match1:
            temp_throughput.append(float(match1.group(1)))
        elif match2:
            temp_response_length.append(float(match2.group(1)))
        elif match3:
            temp_latency.append(float(match3.group(1)))
        elif match4:
            temp_energy.append(float(match4.group(1)))
        
    match_name = False

    throughput.append(temp_throughput.copy())
    response_length.append(temp_response_length.copy())
    latency.append(temp_latency.copy())
    energy.append(temp_energy.copy())

    temp_throughput.clear()
    temp_response_length.clear()
    temp_latency.clear()
    temp_energy.clear()


avg_throughput = [sum(row) / len(row) for row in throughput]
avg_response_length = [sum(row) / len(row) for row in response_length]
avg_latency = [sum(row) / len(row) for row in latency]
avg_energy = [sum(row) / len(row) for row in energy]

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

