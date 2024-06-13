#!/bin/bash

# Define the input names
input_names=("housing_train" "housing_test")

# Get the number of processes from the command line argument
n_proc=$1
echo "Running with $n_proc processes"

# Loop through each input name
for name in "${input_names[@]}"; do
    input_file="data/$name.tsv"

    # Get the total number of lines in the input file
    total_lines=$(wc -l < "$input_file")
    # echo "Total rows: $total_lines"

    # Calculate the number of lines per split file
    lines_per_file=$(( (total_lines + n_proc - 1) / n_proc ))
    # echo "Rows per split: $lines_per_file"

    # Split the input file into smaller files with the calculated number of lines
    split -l $lines_per_file "$input_file" part_

    # Rename the split files
    a=0
    for part in part_*; do
        mv "$part" "data/${name}_$(printf "%01d" $a).tsv"
        a=$((a + 1))
    done
done