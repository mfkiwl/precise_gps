import tensorflow as tf
from src.models.models import *
from src.models.initialization import *

class Prior():
    """
    Add penalty to the marginal log likelihood when optimizing.
    """

    def lasso(self, model) -> tf.Tensor:
        """
        L1 penalty

        Args:
            model (model instance) : src.models.models
        
        Returns:
            tensor
        """
        return -model.lasso*tf.math.reduce_sum(tf.abs(model.kernel.precision()))

    def wishart(self, model) -> tf.Tensor:
        """
        Wishart process penalty

        Args:
            model (model instance) : src.models.models
        
        Returns:
            tensor
        """
        L = tfp.math.fill_triangular(model.kernel.L) # TODO: Checks (not all matrices have L)
        P = model.kernel.precision()
        return (- 1) * tf.math.reduce_sum(tf.math.log(tf.linalg.tensor_diag_part(tf.math.maximum(tf.cast(1e-8, tf.float64), tf.math.abs(L))))) - tf.linalg.trace(1/model.n*tf.eye(model.n, dtype = tf.float64)@P) / 2
        #return (model.n - model.p - 1)/2 * tf.math.log(tf.linalg.det(P)) - tf.linalg.trace(model.V@P) / 2
    
    def horseshow(self, model) -> tf.Tensor:
        """
        Horseshoe penalty

        Args:
            model (model instance) : src.models.models
        
        Returns:
            tensor
        """
        return NotImplementedError
    
    def inverse_wishart(self, model) -> tf.Tensor:
        """
        Wishart process penalty

        Args:
            model (model instance) : src.models.models
        
        Returns:
            tensor
        """
        print(model.kernel.L)
        L = tfp.math.fill_triangular(model.kernel.L) # TODO: Checks (not all matrices have L)
        P = tf.linalg.inv(model.kernel.precision())
        return -(2*model.n + 1) * tf.math.reduce_sum(tf.math.log(tf.linalg.tensor_diag_part(tf.math.maximum(tf.cast(1e-8, tf.float64),tf.math.abs(L))))) - tf.linalg.trace(P) / 2
        #return (model.n - model.p - 1)/2 * tf.math.log(tf.linalg.det(P)) - tf.linalg.trace(model.V@P) / 2