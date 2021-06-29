import argparse, json, pickle, inspect, os 
from src.train import * 
import src.datasets.datasets 
from src.datasets.datasets import *

'''
Training different Gaussian process models and kernels is possible with this script. 
Running isntructions are given in a json file with the following syntax.

{
    "<name>": {
        "model" : (string),
        "kernel": (string),
        "data": (string),
        "lassos": (list),
        "max_iter": (int),
        "num_runs": (int),
        "randomized": (bool),
        "show": (bool),
        "num_Z": (int),
        "minibatch": (int),
        "batch_iter": (int),
        "split": (float)
    }

}
    Args:
        name (string)     : name of the instance
        model (string)    : name of the model (src.models.models)
        kernel (string)   : name of the kernel (src.models.kernels)
        data (string)     : name of the dataset (src.datasets.datasets)
        lassos (list)     : [start, step, end] e.g. [0,0.1,10]
        max_iter (int)    : maximum number of iterations for Scipy
        num_runs (int)    : number of runs per a lasso coefficient
        randomized (bool) : initialization is randomized if True
        show (bool)       : show optimized precisions if True (these are saved anyway)
        num_Z (int)       : number of indusing points
        minibatch (int)   : number of points in minibatch
        batch_iter (int)  : number of iterations for Adam
        split (float)     : test/train split (tells the size of the testset, between 0-1)


See example json-files in "run_files". Results are automatically saved in "results/<name>.pkl".
Usage : python app.py -f <path to json>
'''
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True, help="Path to the json file that is used for running the script.")
args = vars(ap.parse_args())

path = args["file"] #json file containing the commands

# List of possible datasets in src.datasets.datasets
_possible_datasets = list(map (lambda x : x[0], inspect.getmembers(src.datasets.datasets, inspect.isclass)))

def main(path):
    file = open(path,)
    commands = json.load(file)
    file.close()
    for key in commands.keys():
        print(f"Started process for {key}")
        current_run = commands[key]
        model = current_run["model"]
        kernel = current_run["kernel"]

        dataset = current_run["data"]
        if dataset not in _possible_datasets:
            dataset = _possible_datasets[0]
            print(f"Changed to dataset {dataset}")
        else:
            print(f"Using dataset {dataset}")
        
        data_instance = globals()[dataset](current_run["split"])

        l = current_run["lassos"]
        if len(l) == 3:
            lassos = np.arange(l[0], l[2], l[1])
        else:
            lassos = np.array([0])
        
        max_iter = current_run["max_iter"]
        num_runs = current_run["num_runs"]
        randomized  = current_run["randomized"]
        show = current_run["show"]

        num_Z = current_run["num_Z"]
        minibatch_size = current_run["minibatch"]
        batch_iter = current_run["batch_iter"]

        result = train(model, kernel, data_instance, lassos, max_iter, num_runs, randomized, show, num_Z, minibatch_size, batch_iter)

        # Save results
        save_path = f"results/raw/{dataset.lower()}/{os.path.basename(path).split('.')[0]}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        save = open(save_path + f"/{key}.pkl", "wb")
        pickle.dump(result, save)
        save.close()
      
if __name__ == "__main__":
    main(path)