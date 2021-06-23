import os
from app import *

def listdir_nohidden(path):
    '''
    Extension to os.listdir to ignore files starting with '.' or '_'
    '''
    for f in os.listdir(path):
        if not (f.startswith('.') or f.startswith('_')):
            yield f

json_path = "run_files"
run_file_names = [name for name in listdir_nohidden(json_path) if os.path.isdir(os.path.join(json_path, name))]

for run_file in run_file_names:
    path = json_path + "/" + run_file
    print(f"Started process for {path}...")
    main(path)
