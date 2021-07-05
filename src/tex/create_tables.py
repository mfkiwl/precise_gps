import latextable as ltx 
import numpy as np 

def save_eigen_table(data, name, dim, path):
    """
    Save eigenvalues of precision matrix into a latex table.

    Args:
        data (list)   : each index is a list of eigenvalues of a specific iteration
        name (string) : name of the model
        dim (int)     : number of eigenvalues
        path (string) : results saved
    
    Returns:
        tex-file that is saved into path
    """
    table = ltx.Texttable()
    table.set_cols_align(["c"]*(dim+1))
    rows = [['Iter'] + [str(i) for i in range(dim)]]
    for idx, values in enumerate(data):
        ar = [str(idx)] + list(np.round(values,6))
        rows.append(ar)
    table.add_rows(rows)
        
    
    with open(f"{path}{name}.tex", "a") as file:
        file.write(ltx.draw_latex(table, caption=f"{name}"))

def save_results_table(log_liks, train_errors, test_errors, names, path):
    """
    Save results of different models to a tex-file as a table.

    Args:
        log_liks (list)     : log likelihoods for different models (each index is a list)
        train_errors (list) : train errors for different models (each index is a list)
        test_errors (list)  : test errors for different models (each index is a list)
        names (list)        : names of different models 
        path (string)       : results saved
    
    Returns:
        tex-file that is saved into path
    """
    table = ltx.Texttable()
    table.set_cols_align(["c"]*7)
    rows = [["Model", "ll (mean)", "Best ll", "mrmse (train)", "best rmse (train)", "mrmse (test)", "best rmse (test)"]]
    for idx in range(len(names)):
        model = names[idx]
        mean_ll, var_ll = np.round(np.mean(log_liks[idx]),3), np.round(np.std(log_liks[idx]),3)
        max_ll = np.round(np.max(log_liks[idx]),3)

        mean_train, var_train = np.round(np.mean(train_errors[idx]),3), np.round(np.std(train_errors[idx]),3)
        min_train = np.round(np.min(train_errors[idx]),3)

        mean_test, var_test = np.round(np.mean(test_errors[idx]),3), np.round(np.std(test_errors[idx]),3)
        min_test = np.round(np.min(test_errors[idx]),3)
        ar = [f"{model}", f"{mean_ll}({var_ll})", f"{max_ll}", f"{mean_train}({var_train})", f"{min_train}", f"{mean_test}({var_test})", f"{min_test}"]
        rows.append(ar)
    table.add_rows(rows)
        
    with open(f"{path}results.tex", "a") as file:
        file.write(ltx.draw_latex(table, caption=""))