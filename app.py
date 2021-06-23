import argparse
import os 
import json 
from src.train import * 
import pickle 
from src.datasets.datasets import * 

'''
Training different Gaussian process models is possible with this script. Running isntructions are given in
a json file with the following syntax.

{
    "<name>": {
        "model" : (string),
        "kernel": (string),
        "data": (string),
        "lassos": (list),
        "max_iter": (int),
        "num_runs": (int),
        "randomized": (bool),
        "show": (bool)
    }

}
    Args:
        name (string)     : name of the instance
        model (string)    : name of the model ("full", "own_ard", "gpflow_ard")
        kernel (string)   : name of the kernel that is used ("full", "own_ard", "gpflow_ard")
        data (string)     : path to the data e.g. "data/wine/winequality-red.csv"
        lassos (list)     : [start, step, end] e.g. [0,0.1,10]
        max_iter (int)    : maximum number of iterations for Scipy
        num_runs (int)    : number of runs per a lasso coefficient
        randomized (bool) : initialization is randomized if True
        show (bool)       : show optimized precisions if True (these are saved anyway)

See example json-file in "run_files/test.json". Results are automatically saved in "results/<name>.pkl".
Usage : python app.py -f <path to json>
'''
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--file", required=True, help="Path to the json file that is used for running the script.")
args = vars(ap.parse_args())

save_path = "results"
path = args["file"]

possible_datasets = ['Redwine', 'Whitewine', 'Boston', 'Concrete', 'Power', 'Protein', 'Energy', 'Yacht', 'Naval']

def main():
    file = open(path,)
    commands = json.load(file)
    file.close()
    for key in commands.keys():
        print(f"Started process for {key}")
        current_run = commands[key]
        model = current_run["model"]
        kernel = current_run["kernel"]

        dataset = current_run["data"]
        if dataset not in possible_datasets:
            raise NameError(f"{dataset} is not part of the supported datasets:\n{possible_datasets}")
        
        data_instance = globals()[dataset](current_run["split"])
        #data = {}
        #data["train_X"] = data_instance.train_X
        #data["train_y"] = data_instance.train_y
        #data["test_X"] = data_instance.test_X
        #data["test_y"] = data_instance.test_y
        #data["cols"] = data_instance.cols

        l = current_run["lassos"]
        if len(l) == 3:
            lassos = np.arange(l[0], l[2], l[1])
        else:
            lassos = np.array(0)
        
        max_iter = current_run["max_iter"]
        num_runs = current_run["num_runs"]
        randomized  = current_run["randomized"]
        show = current_run["show"]

        num_Z = current_run["num_Z"]
        minibatch_size = current_run["minibatch"]
        batch_iter = current_run["batch_iter"]

        result = train(model, kernel, data_instance, lassos, max_iter, num_runs, randomized, show, num_Z, minibatch_size, batch_iter)

        # Save results
        save = open(f"results/{key}.pkl", "wb")
        pickle.dump(result, save)
        save.close()
      
if __name__ == "__main__":
    main()