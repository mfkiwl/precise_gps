import argparse
import os 
import json 
from src.train import * 
import pickle 
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
import pandas as pd 

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

def main():
    file = open(path,)
    commands = json.load(file)
    file.close()
    for key in commands.keys():
        print(f"Started process for {key}")
        current_run = commands[key]
        model = current_run["model"]
        kernel = current_run["kernel"]

        data_path = current_run["data"]

        if data_path == "data/redwine":
            data_ = genfromtxt(data_path + "/data.csv", delimiter=';')
            Xnp = data_[1:,0:-1]
            for i in range(11):
                Xnp[:,i] -= np.mean(Xnp[:,i])
                Xnp[:,i] /= np.std(Xnp[:,i])
            # scale outputs to [0,1]
            ynp = data_[1:,-1]
            ynp = (ynp - np.min(ynp)) / (np.max(ynp) - np.min(ynp))
        else:
            data_ = genfromtxt(data_path + "/data.txt", delimiter='  ')
            Xnp = data_[:,0:-2]
            Xnp = np.delete(Xnp, [8, 11], axis=1)
            for i in range(16):
                mean = np.mean(Xnp[:,i])
                std = np.std(Xnp[:,i])
                Xnp[:,i] -= mean 
                Xnp[:,i] /= std 

            # scale outputs to [0,1]
            ynp = data_[:,-2]
            ynp = (ynp - np.min(ynp)) / (np.max(ynp) - np.min(ynp))
        # standardize each covariate to mean 0 var 1


        train_Xnp, test_Xnp, train_ynp, test_ynp = train_test_split(Xnp, ynp, test_size=0.20, random_state=42)
        train_data = {}
        train_data["train_X"] = train_Xnp
        train_data["train_y"] = train_ynp
        train_data["test_X"] = test_Xnp
        train_data["test_y"] = test_ynp

        features = [d.strip("'") for d in list(pd.read_csv(data_path + "/features.csv", delimiter=','))]
        train_data["cols"] = features


        l = current_run["lassos"]
        lassos = np.arange(l[0], l[2], l[1])
        
        max_iter = current_run["max_iter"]
        num_runs = current_run["num_runs"]
        randomized  = current_run["randomized"]
        show = current_run["show"]

        num_Z = current_run["num_Z"]
        minibatch_size = current_run["minibatch"]
        batch_iter = current_run["batch_iter"]

        result = train(model, kernel, train_data, lassos, max_iter, num_runs, randomized, show, num_Z, minibatch_size, batch_iter)

        # Save results
        save = open(f"results/{key}.pkl", "wb")
        pickle.dump(result, save)
        save.close()
      
if __name__ == "__main__":
    main()