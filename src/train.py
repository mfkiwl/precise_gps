from models import *
from kernels import *
from utils import * 
from visuals import *
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error


def train(model, kernel, data, lassos, n_splits, max_iter, show = False):


    X = data[0]
    y = data[1]
    cols = data[2]


    kf = KFold(n_splits=n_splits)
    test_errors_full = []
    train_errors_full = []
    mlls = {}
    params = {}
    for l in lassos:
        mlls[l] = {}
        params[l] = {}
        errors = []
        train_errors = []
        counter = 0
        for train_idx, test_idx in kf.split(X):
            mlls[l][counter] = []
            params[l][counter] = []
            train_Xnp, train_ynp = tf.gather(X,train_idx), tf.gather(y,train_idx)
            test_Xnp, test_ynp = tf.gather(X,test_idx), tf.gather(y,test_idx)
            if model == "FULL":
                kernel = FullGaussianKernelCholesky(10,1,1,True)
                gpr_model = GPRLassoFullCholesky((X,y),kernel, l)
            else:
                kernel = ARD(10,1,1,True)
                gpr_model = GPRLassoARD((X,y),kernel, l)
            
            optimizer = gpflow.optimizers.Scipy()
            if model == "FULL":
                def step_callback(step, variables, values):
                    paddings = tf.constant([[1, 0,], [0, 1]])
                    L_as_matrix = tfp.math.fill_triangular(gpr_model.kernel.off_diagonal)
                    L_as_matrix = tf.pad(L_as_matrix, paddings, "CONSTANT")
                    L = tf.linalg.set_diag(L_as_matrix, gpr_model.kernel.diagonal)
                    params[l][counter].append(list(L))
                    value = gpr_model.lml_lasso()
                    mlls[l][counter].append(value)
                    if step % 100 == 0:
                        print(f"Step {step}, MLL: {value.numpy()}")
            else:
                def step_callback(step, variables, values):
                    P = tf.linalg.diag(kernel.lengthscales)
                    params[l][counter].append(list(P))
                    value = gpr_model.lml_lasso()
                    mlls[l][counter].append(value)
                    if step % 100 == 0:
                        print(f"Step {step}, MLL: {value.numpy()}")
            
            optimizer.minimize(
                gpr_model.training_loss, gpr_model.trainable_variables, options={'maxiter': max_iter,'disp': True}, step_callback = step_callback)

        # Calculating error
        pred_mean, _ = gpr_model.predict_y(test_Xnp)
        pred_train,_ = gpr_model.predict_y(train_Xnp)
        rms = mean_squared_error(test_ynp.numpy(), pred_mean.numpy(), squared=False)
        rms_train = mean_squared_error(train_ynp.numpy(), pred_train.numpy(), squared=False)
        errors.append(rms)
        train_errors.append(rms_train)

        if show:
            if model == "FULL":
                paddings = tf.constant([[1, 0,], [0, 1]])
                L_as_matrix = tfp.math.fill_triangular(gpr_model.kernel.off_diagonal)
                L_as_matrix = tf.pad(L_as_matrix, paddings, "CONSTANT")
                L = tf.linalg.set_diag(L_as_matrix, gpr_model.kernel.diagonal)
                show_kernel(L @ tf.transpose(L), "Optimized precision matrix LL^T", cols, "", "center")
            else:
                P = tf.linalg.diag(kernel.lengthscales)
                show_kernel(P, "Optimized precision matrix LL^T", cols, "", "center")

        counter += 1
        current_mean = np.mean(errors)
        train_mean = np.mean(train_errors)
        print("Train error:", train_mean, "Validation error", current_mean)
        test_errors_full.append(current_mean)
        train_errors_full.append(train_mean)

    df = {}
    df["data"] = data
    df["lassos"] = lassos
    df["max_iter"] = max_iter 
    df["test_errors"] = test_errors_full
    df["train_errors"] = train_errors_full
    df["mll"] = parse_trace(mlls, lassos, max_iter = max_iter)
    df["params"] = parse_traceL(params, lassos, max_iter = max_iter)


