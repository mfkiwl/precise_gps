import os 
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd 
from src.parse_results import parse_pickle
from src.visuals.visuals import * 
from src.datasets.datasets import *
from src.visuals.process_results import *

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
    visualize_errors(train_errors,names,"train",result_path + "/train_errors.pdf")
    visualize_errors(test_errors,names,"test",result_path + "/test_errors.pdf")

    # Kernels
    for key in df.keys():
        data = df[key]
        for l in data["lassos"]:
            precisions = []
            p_names = []
            for i in range(9):
                if data["kernel"] == "FullGaussianKernel":
                    L = data["params"][l][i][-1]
                    L = tfp.math.fill_triangular(L)
                    precisions.append(L @ tf.transpose(L))
                    p_names.append("LL: " + str(round(data["log_likelihoods"][l][i].numpy(),2)) +" TE: " + str(round(data["test_errors"][l][i],2)))
                else:
                    precisions.append(tf.linalg.diag(np.array(data["params"][l][i][-1])**(-2)))
                    p_names.append("LL: " + str(round(data["log_likelihoods"][l][i].numpy(),2)) +" TE: " + str(round(data["test_errors"][l][i],2)))
            data_instance = globals()[dataset](0.2)
            cols = data_instance.cols
            if not os.path.exists(result_path + "/kernels"):
                os.makedirs(result_path + "/kernels")

            show_kernels(precisions,p_names,cols,"own",-1,result_path + "/kernels" + "/" + data["model"] + data["kernel"] + str(l) + ".pdf")

    # Eigenvalues
    for key in df.keys():
        data = df[key]
        eigen_dataframe = {}
        ret = pd.DataFrame(data=eigen_dataframe)
        for l in data["lassos"]:
            for i in range(10):
                if data["kernel"] == "FullGaussianKernel":
                    L = data["params"][l][i][-1]
                    L = tfp.math.fill_triangular(L)
                    eigen_vals = eigen(L@tf.transpose(L))
                    ret[(l,i)] = list(eigen_vals)
        if not os.path.exists(result_path + "/eigen"):
            os.makedirs(result_path + "/eigen")
        ret.to_csv(result_path + "/eigen/{}_{}".format(data["model"], data["kernel"]))




    # Loss landscape
    if not os.path.exists(result_path + "/loss_landscape"):
        os.makedirs(result_path + "/loss_landscape")
    for key in df.keys():
        data = df[key]
        if data["kernel"] == "FullGaussianKernel":
            for l in data["lassos"]:
                visualize_loss_landscape(data, data["model"], data["kernel"], data["data_train"], l, True,10, result_path + "/loss_landscape/" + "{}_{}_{}".format(data["model"], data["kernel"], l))

