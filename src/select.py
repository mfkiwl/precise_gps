import inspect, importlib 
import src.models.kernels
import src.models.models
import src.datasets.datasets

_defined_kernels = inspect.getmembers(src.models.kernels, inspect.isclass)
_defined_models = inspect.getmembers(src.models.models, inspect.isclass)
_defined_datasets = inspect.getmembers(src.datasets.datasets, inspect.isclass)

_possible_kernels = [x[0] for x in _defined_kernels 
                     if x[1].__module__ == 'src.models.kernels']
_possible_models = [x[0] for x in _defined_models 
                    if x[1].__module__ == 'src.models.models']
_possible_datasets = [x[0] for x in _defined_datasets 
                      if x[1].__module__ == 'src.datasets.datasets']

def select_kernel(kernel, **kwargs):
    '''
    Create instance of a kernel, that is from src.models.kernels.

    Args:
        kernel (str) : name of the class (class.__name__)
        kwargs (dict) : must containt the required init variables of the 
        kernel (see src.models.kernels)
    
    Returns:
        Kernel from src.models.kernels
    '''
    if kernel not in _possible_kernels:
        kernel = _possible_kernels[0]
        print(f'Changed to kernel {kernel}.')
    else:
        print(f'Using kernel {kernel}.')

    module = importlib.import_module('src.models.kernels')
    kernel_instance = getattr(module, kernel)(**kwargs)

    return kernel_instance 

def select_model(model, **kwargs):
    '''
    Create instance of a model, that is from src.models.models.

    Args:
        model (string) : name of the class (class.__name__)
        kwargs (dict)  : must containt the required init variables of 
        the model (see src.models.models)
    
    Returns:
        Model from src.models.models
    '''
    if model not in _possible_models:
        model = _possible_models[0]
        print(f'Changed to model {model}.')
    else:
        print(f'Using model {model}.')

    module = importlib.import_module('src.models.models')
    model_instance = getattr(module, model)(**kwargs)

    return model_instance

def select_dataset(dataset, split):
    '''
    Create instance of a dataset, that is from src.datasets.datasets.

    Args:
        dataset (string) : name of the class (class.__name__)
        kwargs (dict)  : must containt the required init variables of 
        the dataset (see src.datasets.datasets)
    
    Returns:
        Dataset from src.datasets.datasets
    '''
    if dataset not in _possible_datasets:
        dataset = _possible_datasets[0]
        print(f'Changed to dataset {dataset}.')
    else:
        print(f'Using dataset {dataset}.')

    module = importlib.import_module('src.datasets.datasets')
    dataset_instance = getattr(module, dataset)(split)

    return dataset_instance   