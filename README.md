# Precise Gaussian processes

The goal of this project is to try to exploit the full potential of the covariance matrix in Gaussian processes, mainly, the motivation is to learn the off-diagonal couplings of the covariance matrix. The argument is that this is going to improve interpretability - once the combination with predictive capacity is found. Further on, sparse covariate structures are going to simplify the model. We also know that in many applications a true covariate structure exists, for example, different parts of a robot are connected to each other. 

## Usage

Training models is possible by running the training script with a specified file containing the inputs

```
python app.py -f <path to file>
```

