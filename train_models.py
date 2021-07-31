import argparse, json, pickle, os 

from src.train import train
from src.datasets.datasets import *
from src.select import select_dataset

'''
Train different Gaussian process models. Current implementations include
standard Gaussian process regression, stochastic variational inference, 
and stochastic gradient hamiltonian monte carlo. Running instructions 
are given in a json file with the following syntax.

{
    '<name>': {
        'model' : (str),
        'kernel': (str),
        'data': (str),
        'lassos': (list),
        'max_iter': (int),
        'num_runs': (int),
        'randomized': (bool),
        'show': (bool),
        'num_Z': (int),
        'minibatch': (int),
        'batch_iter': (int),
        'split': (float),
        'rank': (int),
        'penalty': (str)
    }

}
    Args:
        name (str) (required) : name of the instance
        model (str) (required) : name of the model
        kernel (str) (required) : name of the kernel
        data (str) (required) : name of the dataset
        penalty (str) (optional) : name of the penalty used
        lassos (list) (optional) : [start, step, end]
        max_iter (int) (optional) : iterations for Scipy
        num_runs (int) (optional) : number of initializations
        randomized (bool) (optional) : wheter randomized init is used
        show (bool) (optional) : show optimized precisions if True
        num_Z (int) (optional) : number of indusing points
        minibatch (int) (optional) : number of points in minibatch
        batch_iter (int) (optional) : number of iterations for Adam
        split (float) (optional) : test/train split (testset size)
        rank (int) (optional) : rank of the precision matrix
        n (int) (optional) : wishart degrees of freedom
        V (list) (optional) : wishart scale matrix

Required arguments are name, model, kernel, and data. See example 
json-files in 'run_files'. Results are automatically saved in 
'results/raw/<name>.pkl'. Usage : python app.py -f <path to json>
'''
ap = argparse.ArgumentParser()
ap.add_argument('-f', '--file', required=True, help='Path to the json-file.')
args = vars(ap.parse_args())

path = args['file'] # json-file containing the commands
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PENALTY, LASSOS, MAX_ITER, NUM_RUNS, RANDOMIZED = 'lasso', [0], 1000, 10, 1
NUM_Z, MINIBATCH_SIZE, BATCH_ITER, RANK = 100, 100, 40_000, 1

def main(path):
    '''
    Trains model specified in the json-file, and saves the results into
    a pickle-file.
    
    Args:
        path (str) : path to the json-file
    '''
    with open(path,) as file:
        commands = json.load(file)

    for key in commands.keys():
        print(f'Started process for {key}!')
        current_run = commands[key]

        model = current_run['model']
        kernel = current_run['kernel']
        dataset = current_run['data']
        data_instance = select_dataset(dataset, current_run['split'])

        # Select lasso coefficients
        lasso_input = LASSOS if 'lassos' not in current_run else \
                      current_run['lassos']
        
        if len(lasso_input) == 3:
            lassos = np.arange(lasso_input[0], lasso_input[2], 
                               lasso_input[1])
        else:
            lassos = np.array([0])

        # Select n for Wishart
        n_input = [data_instance.train_X.shape[1]] if 'n' not in current_run \
                   else current_run['n']
                   
        if len(n_input) == 3:
            n = np.arange(n_input[0], n_input[2], n_input[1])
        else:
            n = n_input
        
        # Select other parameters
        penalty = PENALTY if 'penalty' not in current_run else \
                  current_run['penalty']
        max_iter = MAX_ITER if 'max_iter' not in current_run else \
                   current_run['max_iter']
        num_runs = NUM_RUNS if 'num_runs' not in current_run else \
                   current_run['num_runs']
        randomized  = RANDOMIZED if 'randomized' not in current_run else \
                      current_run['randomized']
        num_Z = NUM_Z if 'num_Z' not in current_run else \
                current_run['num_Z']
        minibatch_size = MINIBATCH_SIZE if 'minibatch' else \
                         current_run['minibatch']
        batch_iter = BATCH_ITER if 'batch_iter' not in current_run else \
                     current_run['batch_iter']
        rank = RANK if 'rank' not in current_run else \
               current_run['rank'] 
        V = None if 'V' not in current_run else current_run['V']

        result = train(model, kernel, data_instance, lassos, max_iter, 
                       num_runs, randomized, num_Z, minibatch_size, 
                       batch_iter, rank, penalty, n, V)

        instance_path = os.path.basename(path).split('.')[0]
        save_path = f'results/raw/{dataset.lower()}{instance_path}'
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        with open(f'{save_path}/{key}.pkl', 'wb') as save:
            pickle.dump(result, save)
      
if __name__ == '__main__':
    main(path)