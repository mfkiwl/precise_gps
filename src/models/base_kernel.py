import gpflow
import tensorflow as tf

class BaseKernel(gpflow.kernels.Kernel):
    """
    Own implementation of the squared exponential kernel with ard property. Should work
    the same way as gpflow.kernels.SquaredExponential(ARD = True). Lengthscales and variance 
    can be randomized. This should be handled when initializing the kernel.
    See : https://gpflow.readthedocs.io/en/master/gpflow/kernels/index.html#gpflow-kernels-squaredexponential

    Args:
        variance (float)           : signal variance which scales the whole kernel
        lengthscales (numpy array) : list of lengthscales (should match the dimension of the input)
    """
    def precision(self) -> tf.Tensor:
        return NotImplementedError