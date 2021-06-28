import os 
import tensorflow as tf
import tensorflow_probability as tfp 
from src.parse_results import parse_pickle
from src.visuals.visuals import * 
from src.datasets.datasets import *

plt.rcParams.update({'font.size': 14}) # Global fontsize

def create_visuals(dataset, directory, num_lassos, step = 1):

    data_path = f"results/raw/{dataset.lower()}/{directory}/"
    pkl_files = [file for file in os.listdir(data_path) if '.pkl' in file] # Extract only pickle files

    df = {}
    for idx, current_file in enumerate(pkl_files):
        data = parse_pickle(data_path + current_file)
        df[idx] = data
    
    dataset = data["dataset"]

    result_path = "results/processed/" + f"{dataset.lower()}/{directory}" # current_file.split('.')[0]
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    
    #MLLS, log-likelihoods, and errors
    names = []
    mll_names = []
    mlls = []
    log_liks = []
    train_errors = []
    test_errors = []
    all_lassos = []
    for key in df.keys():
        data = df[key]
        model = data["model"]
        kernel = data["kernel"]

        if "ARD" in kernel:
            kernel = "ARD"
        else:
            kernel = "FULL"
        
        if "GPR" in model:
            model = "GPR"
        else:
            model = "SVI"

        names.append(model + " " + kernel)
        lassos = data["lassos"][0::step]
        lassos = lassos[0:min(len(lassos), num_lassos)]
        for l in lassos:
            mlls.append(data["mll"][l])
            mll_names.append(model + " " + kernel + " " + str(l))
        log_liks.append(data["log_likelihoods"])
        train_errors.append(data["train_errors"])
        test_errors.append(data["test_errors"])
        all_lassos.append(data["lassos"])
    
    visualize_mlls(mlls, mll_names, result_path + "/mlls.pdf")
    visualize_log_likelihood(log_liks, names, result_path + "/log_liks.pdf")
    visualize_errors(train_errors,names,"train","train_errors.pdf")
    visualize_errors(test_errors,names,"test","test_errors.pdf")

    # Kernels
    for key in df.keys():
        data = df[key]
        lassos = data["lassos"][0::step]
        lassos = lassos[0:min(len(lassos), num_lassos)]
        for l in lassos:
            precisions = []
            p_names = []
            for i in range(9):
                print(data["kernel"])
                if data["kernel"] == "FullGaussianKernel":
                    L = data["params"][l][i][-1]
                    print("L:", L.shape)
                    L = tfp.math.fill_triangular(L)
                    precisions.append(L @ tf.transpose(L))
                    p_names.append(data["kernel"] + " " + str(l))
                else:
                    print(data["params"][l][i][-1])
                    precisions.append(data["params"][l][i][-1][0])
                    p_names.append(data["kernel"] + " " + str(l))
            data_instance = globals()[dataset](0.2)
            cols = data_instance.cols
            print(len(precisions), len(p_names))
            show_kernels(precisions,p_names,cols,"global",-1,data["kernel"] + str(l) + ".pdf")

