#!/bin/bash
# Run app.py for multiple json-files (this way files don't become too long).
# Args:
#   path : path to directory containing the json-files

cd $1 # This should be the directory to the json-files
files=$(ls)
cd - 
for file in $files
do
    echo "Startin process for $file ..."
    python app.py -f "$1/$file"
done
