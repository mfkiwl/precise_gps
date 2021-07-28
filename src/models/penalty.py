import tensorflow as tf
from src.models.models import *
from src.models.initialization import *

class Penalty():
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
        return (model.n - model.p - 1) * tf.math.reduce_sum(tf.math.log(tf.linalg.tensor_diag_part(tf.math.maximum(1e-8,tf.math.abs(L))))) - tf.linalg.trace(1/model.n*model.V@P) / 2
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