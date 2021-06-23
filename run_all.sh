#!/bin/bash
cd run_files
files=$(ls)
cd ..
for file in $files
do
    echo "Startin process for $file ..."
    python app.py -f "$file"
done
