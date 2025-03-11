# Define the file path
file_path = "Motivation-13100086.out"

# Define prefixes to filter out
prefixes_to_remove = ["Requirement already satisfied:", "[LightGBM]", "You can set", "And if memory"]

# Read the file and filter lines
with open(file_path, "r") as file:
    lines = file.readlines()

filtered_lines = [
    line for line in lines
    if not any(line.startswith(prefix) for prefix in prefixes_to_remove)
]

# Write the filtered lines back to the same file
with open(file_path, "w") as file:
    file.writelines(filtered_lines)

print("File has been updated successfully.")
