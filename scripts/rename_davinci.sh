#!/bin/bash

# Set the folder path where you want to perform the operation
folder_path="../data/xsum"
# folder_path="../data/writing"
# folder_path="../data/pubmed"

# List all files in the folder that contain "davinci"
matching_files=$(find "$folder_path" -type f -name "*davinci*")

# Loop through each matching file and rename it
for file in $matching_files; do
    new_name=$(echo "$file" | sed 's/davinci/gpt-3/')
    mv "$file" "$new_name"
done