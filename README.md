# Precise Gaussian processes

The goal of this project is to try to exploit the full potential of the covariance matrix in Gaussian processes, mainly, the motivation is to learn the off-diagonal couplings of the covariance matrix. The argument is that this is going to improve interpretability - once the combination with predictive capacity is found. Further on, sparse covariate structures are going to simplify the model. We also know that in many applications a true covariate structure exists, for example, different parts of a robot are connected to each other. 

## Usage

Training models is possible by running the training script with a specified json-file containing the instructions

```
python app.py -f <path to json file>
```

Running the training script requires a `-f` or `--file` command. We have to provide a json-file which provides the instructions for the training. The json-file has the following format. 

```json
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
```

Here name denotes the training instance. For further explanation see `app.py`. For example, the file for running could look something like this. 

```json
{
    "test1": {
        "model" : "full",
        "kernel": "full",
        "data": "data/wine/winequality-red.csv",
        "lassos": [0,0.1,1],
        "max_iter": 1000,
        "num_runs": 5,
        "randomized": 1,
        "show": 0
    },

    "test2": {
        "model" : "full",
        "kernel": "full",
        "data": "data/wine/winequality-red.csv",
        "lassos": [0,0.1,1],
        "max_iter": 500,
        "num_runs": 10,
        "randomized": 0,
        "show": 1
    }
}
```
We are able to run multiple separate instances with one run command. The results are saved into `results/<name>.pkl`, so in the above case two files would be created: `results/test1.pkl` and `results/test2.pkl`. One sample json-file is also provided in `run_files/test.json`. 


