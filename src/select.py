import inspect, importlib 
import src.models.kernels

_possible_kernels = []
kernels = inspect.getmembers(src.models.kernels, inspect.isclass)
for k in kernels:
    _possible_kernels.append(k[0][0])


import src.models.models
_possible_models = []
models = inspect.getmembers(src.models.models, inspect.isclass)
for m in models:
    _possible_models.append(m[0][0])

def select_kernel(kernel, **kwargs):
    print(_possible_kernels)
    if kernel not in _possible_kernels:
        kernel = _possible_kernels[0]

    module = importlib.import_module("src.models.kernels")
    kernel_instance = getattr(module, kernel)(**kwargs)

    return kernel_instance 


def select_model(model, **kwargs):
    if model not in _possible_models:
        model = _possible_models[0]

    module = importlib.import_module("src.models.models")
    model_instance = getattr(module, model)(**kwargs)

    return model_instance 