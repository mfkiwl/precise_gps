import inspect, importlib 
import src.models.kernels
import src.models.models
import src.datasets.datasets

# Private lists for kernels and models defined in src.models 
_possible_kernels = list(map (lambda x : x[0], inspect.getmembers(src.models.kernels, inspect.isclass)))
_possible_models = list(map (lambda x : x[0], inspect.getmembers(src.models.models, inspect.isclass)))
_possible_datasets = list(map (lambda x : x[0], inspect.getmembers(src.datasets.datasets, inspect.isclass)))

def select_kernel(kernel, **kwargs):
    """
    Create instance of a kernel, that is from src.models.kernels.

    Args:
        kernel (string) : name of the class (class.__name__)
        kwargs (dict)   : must containt the required init variables of the kernel (see src.models.kernels)
    
    Returns:
        Class instance of the kernel
    """
    if kernel not in _possible_kernels:
        kernel = _possible_kernels[0]
        print(f"Changed to kernel {kernel}.")
    else:
        print(f"Using kernel {kernel}.")

    module = importlib.import_module("src.models.kernels")
    kernel_instance = getattr(module, kernel)(**kwargs)

    return kernel_instance 

def select_model(model, **kwargs):
    """
    Create instance of a model, that is from src.models.models.

    Args:
        model (string) : name of the class (class.__name__)
        kwargs (dict)  : must containt the required init variables of the model (see src.models.models)
    
    Returns:
        Class instance of the model
    """
    if model not in _possible_models:
        model = _possible_models[0]
        print(f"Changed to model {model}.")
    else:
        print(f"Using model {model}.")

    module = importlib.import_module("src.models.models")
    model_instance = getattr(module, model)(**kwargs)

    return model_instance

def select_dataset(dataset, split):
    """
    Create instance of a dataset, that is from src.datasets.datasets.

    Args:
        dataset (string) : name of the class (class.__name__)
        kwargs (dict)  : must containt the required init variables of the dataset (see src.datasets.datasets)
    
    Returns:
        Class instance of the dataset
    """
    if dataset not in _possible_datasets:
        dataset = _possible_models[0]
        print(f"Changed to dataset {dataset}.")
    else:
        print(f"Using dataset {dataset}.")

    module = importlib.import_module("src.datasets.datasets")
    dataset_instance = getattr(module, dataset)(split)

    return dataset_instance   