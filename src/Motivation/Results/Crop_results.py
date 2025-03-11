from glob import glob
import os

file = 'results.txt'
cropped_file = 'results_cropped_intermediate.txt'
result_file = 'results_cropped.txt'

with open(file, 'r') as f:
    lines = f.readlines()
    for line in lines:
        if line.startswith("Dataset:"):
            with open(cropped_file, 'a') as c:
                c.write(line)
        if line.__contains__("Results: "):
            with open(cropped_file, 'a') as c:
                c.write(line)
        if line.startswith("0 "):
            with open(cropped_file, 'a') as c:
                c.write(line)
        if line.startswith("1 "):
            with open(cropped_file, 'a') as c:
                c.write(line)

with open(cropped_file, 'r') as c:
    lines = c.readlines()
    with open(result_file, 'a') as r:
        for i in range(len(lines)):
            if lines[i].startswith(("0", "1")):
                if i >= 6 and not any("Results" in lines[i - j] for j in range(1, 7)):
                    print(lines[i]) # Debugging output
                else:
                    r.write(lines[i]) # Write non-filtered lines
            elif "Dataset" in lines[i] and any("Dataset" in lines[i - j] for j in range(1, min(i, 3) + 1)):
                print(lines[i])  # Debugging output
            else:
                r.write(lines[i])  # Write non-filtered lines

# os.remove(cropped_file)