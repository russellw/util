import csv

# Input and output file paths
input_file = "input.txt"
output_file = "output.csv"

# Read the input text file and split it into lines
with open(input_file, "r") as infile:
    lines = infile.read().splitlines()

# Make sure the number of lines is a multiple of 4 (4N)
if len(lines) % 4 != 0:
    print("Error: The number of lines in the input file is not a multiple of 4.")
    exit(1)

# Create a list to store rows of data
rows = []

# Loop through the lines in groups of 4 and append them as a row
for i in range(0, len(lines), 4):
    row = [lines[i], lines[i + 1], lines[i + 2], lines[i + 3]]
    rows.append(row)

# Write the data to the output CSV file
with open(output_file, "w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerows(rows)

print(f"Conversion completed. {len(rows)} rows were written to {output_file}.")
