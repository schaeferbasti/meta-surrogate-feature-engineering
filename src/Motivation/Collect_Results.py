import os

files = os.listdir("../../results_*.txt")
for file in files:
    with open(file, 'r') as f:
        content = f.read()
        with open('results.txt', 'a') as results_file:
            results_file.write(content)
