#!/bin/bash

# Set the folder path where you want to perform the operation
# folder_path="../data/xsum"
# folder_path="../data/writing"
folder_path="../data/pubmed"

# List all files in the folder that contain "gpt-3.5-turbo"
matching_files=$(find "$folder_path" -type f -name "*gpt-3.5-turbo*")

# Loop through each matching file and rename it
for file in $matching_files; do
    new_name=$(echo "$file" | sed 's/-turbo//')
    mv "$file" "$new_name"
done