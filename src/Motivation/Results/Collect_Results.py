from glob import glob
import os

files = glob('../../results_*.txt')
print(files)

with open('../../results.txt', 'w') as results_file:
    results_file.write("Collected Results\n")
for file in files:
    with open(file, 'r') as f:
        content = f.read()
        with open('../../results.txt', 'a') as results_file:
            results_file.write(content)
    os.remove(file)