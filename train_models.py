import argparse, json, pickle, os 
from src.train import * 
from src.datasets.datasets import *
from src.select import select_dataset

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
        "rank": (int)
    }

}
    Args:
        name (string) (required)     : name of the instance
        model (string) (required)    : name of the model (src.models.models)
        kernel (string) (required)   : name of the kernel (src.models.kernels)
        data (string) (required)     : name of the dataset (src.datasets.datasets)
        penalty (string) (optional)  : name of the penalty used (src.models.penalty)
        lassos (list) (optional)     : [start, step, end] e.g. [0,0.1,10]
        max_iter (int) (optional)    : maximum number of iterations for Scipy
        num_runs (int) (optional)    : number of runs per a lasso coefficient
        randomized (bool) (optional) : initialization is randomized if True
        show (bool) (optional)       : show optimized precisions if True (these are saved anyway)
        num_Z (int) (optional)       : number of indusing points
        minibatch (int) (optional)   : number of points in minibatch
        batch_iter (int) (optional)  : number of iterations for Adam
        split (float) (optional)     : test/train split (tells the size of the testset, between 0-1)
        rank (int) (optional)        : specifies the rank of the precision matrix
        n (int) (optional)           : wishart degrees of freedom
        V (list) (optional)          : wishart process V matrix

Required arguments are name, model, kernel, and data. See example json-files in "run_files". Results are automatically saved in "results/raw/<name>.pkl".
Usage : python app.py -f <path to json>
'''
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True, help="Path to the json file that is used for running the script.")
args = vars(ap.parse_args())

path = args["file"] # json-file containing the commands

# Default values if not given
PENALTY = "lasso"
LASSOS = [0]
MAX_ITER = 1000
NUM_RUNS = 10
RANDOMIZED = 1
NUM_Z = 100
MINIBATCH_SIZE = 100
BATCH_ITER = 40_000
RANK = 1

def main(path):
    # Read data from file
    with open(path,) as file:
        commands = json.load(file)

    for key in commands.keys():
        print(f"Started process for {key}!")
        current_run = commands[key]

        # Model, kernel and dataset
        model = current_run["model"]
        kernel = current_run["kernel"]
        dataset = current_run["data"]
        data_instance = select_dataset(dataset, current_run["split"])

        # Select lasso coefs
        lasso_input = LASSOS if "lassos" not in current_run else current_run["lassos"]
        if len(lasso_input) == 3:
            lassos = np.arange(lasso_input[0], lasso_input[2], lasso_input[1])
        else:
            lassos = np.array([0])

        # Select n for Wishart
        n_input = [data_instance.train_X.shape[1]] if "n" not in current_run else current_run["n"]
        if len(n_input) == 3:
            n = np.arange(n_input[0], n_input[2], n_input[1])
        
        # Select other parameters
        penalty = PENALTY if "penalty" not in current_run else current_run["penalty"]
        max_iter = MAX_ITER if "max_iter" not in current_run else current_run["max_iter"]
        num_runs = NUM_RUNS if "num_runs" not in current_run else current_run["num_runs"]
        randomized  = RANDOMIZED if "randomized" not in current_run else current_run["randomized"]
        num_Z = NUM_Z if "num_Z" not in current_run else current_run["num_Z"]
        minibatch_size = MINIBATCH_SIZE if "minibatch" else current_run["minibatch"]
        batch_iter = BATCH_ITER if "batch_iter" not in current_run else current_run["batch_iter"]
        rank = RANK if "rank" not in current_run else current_run["rank"] 
        V = None if "V" not in current_run else current_run["V"]

        # Train model
        result = train(model, kernel, data_instance, lassos, max_iter, num_runs, randomized, num_Z, minibatch_size, batch_iter, rank, penalty, n, V)

        # Save results
        save_path = f"results/raw/{dataset.lower()}/{os.path.basename(path).split('.')[0]}"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(save_path + f"/{key}.pkl", "wb") as save:
            pickle.dump(result, save)
      
if __name__ == "__main__":
    main(path)