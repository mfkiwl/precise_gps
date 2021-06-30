import os 
import tensorflow as tf
import tensorflow_probability as tfp
import pandas as pd 
from src.parse_results import parse_pickle
from src.visuals.visuals import * 
from src.datasets.datasets import *
from src.visuals.process_results import *

plt.rcParams.update({'font.size': 14}) # Global fontsize

# TODO: this needs cleaning up
# TODO:
# TODO:

def create_results(dataset, directory, num_lassos, step = 1):
    """
    Create visualizations form the raw .pkl files. This function also forms some
    dataframes. 

    Args:
        dataset (string)   : path to the dataset folder
        directory (string) : dataset directory for analysis
        num_lassos (int)   : number of lasso coefficients (used for some visualizations)
        step (int)         : step between the lasso coefficients
    
    Returns:
        Saves visualizations and dataframes to results/processed
    """

    data_path = f"results/raw/{dataset.lower()}/{directory}/"
    pkl_files = [file for file in os.listdir(data_path) if '.pkl' in file] # Extract only pickle files

    df = {}
    for idx, current_file in enumerate(pkl_files):
        data = parse_pickle(data_path + current_file)
        df[idx] = data
    
    dataset = data["dataset"]

    result_path = "results/processed/" + f"{dataset.lower()}/{directory}"
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
    precisions = []
    all_precisions = []
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
        
        for l in data["lassos"]:
            new_params = {}
            for i in range(10):
                new_params[i] = params_to_precision(np.array(data["params"][l][i][-1]), data["kernel"])
            precisions.append(new_params) 
        
        all_precisions.append(precisions)
        log_liks.append(data["log_likelihoods"])
        train_errors.append(data["train_errors"])
        test_errors.append(data["test_errors"])
        all_lassos.append(data["lassos"])
    
    visualize_mlls(mlls, mll_names, result_path + "/mlls.pdf")
    visualize_log_likelihood(log_liks, names, all_precisions, 10, True, result_path + "/log_liks.pdf")
    visualize_errors(train_errors,names,"train", all_precisions, 10, True, result_path + "/train_errors.pdf")
    visualize_errors(test_errors,names,"test", all_precisions, 10, True, result_path + "/test_errors.pdf")

    # Kernels
    for key in df.keys():
        data = df[key]
        for l in data["lassos"]:
            precisions = []
            p_names = []
            for i in range(9):
                P = params_to_precision(np.array(data["params"][l][i][-1]), data["kernel"])
                precisions.append(P)
                p_names.append("LL: " + str(round(data["log_likelihoods"][l][i].numpy(),2)) +", TE: " + str(round(data["test_errors"][l][i],2)) + ", Var: " + str(np.round(data["variances"][l][i][-1],2)))
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
                P = params_to_precision(np.array(data["params"][l][i][-1]), data["kernel"])
                eigen_vals, _ = eigen(P)
                ret[(l,i)] = list(eigen_vals)
        if not os.path.exists(result_path + "/eigen"):
            os.makedirs(result_path + "/eigen")
        ret.to_csv(result_path + "/eigen/{}_{}.csv".format(data["model"], data["kernel"]))




    # Loss landscape
    if not os.path.exists(result_path + "/loss_landscape"):
        os.makedirs(result_path + "/loss_landscape")
    for key in df.keys():
        data = df[key]
        if data["kernel"] == "FullGaussianKernel":
            for l in data["lassos"]:
                visualize_loss_landscape(data, data["model"], data["kernel"], data["data_train"], l, False,10, result_path + "/loss_landscape/" + "{}_{}_{}.pdf".format(data["model"], data["kernel"], l))

