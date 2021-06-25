# Precise Gaussian processes

The goal of this project is to try to exploit the full potential of the covariance matrix in Gaussian processes, mainly, the motivation is to learn the off-diagonal couplings of the precision matrix. The argument is that this is going to improve interpretability - once the combination with predictive capacity is found. Further on, sparse covariate structures are going to simplify the model. We also know that in many applications a true covariate structure exists, for example, different parts of a robot are connected to each other. 

## Usage

Training  different models is possible by running the training script with a specified json-file containing the instructions

```
python app.py -f <path to json file>
```

Running the training script requires a `-f` or `--file` command since we have to provide a json-file which gives the instructions for the training (what model to use etc.). The json-file has the following format. 

```jsonc
{
    "<name>": {
        "model" : "GPRLasso", // string, possible models in src.models.models
        "kernel": "FullGaussianKernel", // string, possible kernels in src.models.kernels
        "data": "Redwine", // string, possible datasets in src.datasets.datasets
        "lassos": [0,0.1,1], // list, where notation is [start, step, end]
        "max_iter": 1000, // int, max number of iterations for Scipy optimizer
        "num_runs": 5, // int, number of runs with the same initializations
        "randomized": 1, // bool, initialization randomized if True
        "show": 0, // bool, shows intermediate plot during process if True
        "num_Z": 100, // int, number of inducing points
        "minibatch": 100, // int, number datapoints in minibatch
        "batch_iter": 5000, // int, number of iterations for Adam optimizer
        "split": 0.2 // float, test/train split where split indicates the testset size (between 0-1)
    }
}
```

Here `<name>` denotes the training instance, and it can be any string. For further explanation see `app.py`, also, you can check json-file examples from `run_files`. We are able to run multiple separate instances with one run command. The results are saved into `results/raw/<name>.pkl`. If you want to run many json-files at the same time, see `run_all.sh`. You can create a folder with many json-files (like `run_files`).

```
chmod +x run_all.sh
./run_all.sh <path to folder>
```


