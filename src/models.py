import gpflow
import tensorflow as tf 
import tensorflow_probability as tfp 

class GPRLassoFull(gpflow.models.GPR):
    
    def __init__(self, data, kernel, lasso):
        super(GPRLassoFull, self).__init__(data, kernel)
        self.lasso = lasso
    
    def lml_lasso(self):
        return self.log_marginal_likelihood() - self.lasso*tf.math.reduce_sum(tf.abs(self.kernel.L @ tf.transpose(self.kernel.L)))

    def maximum_log_likelihood_objective(self):
        return self.lml_lasso()

class GPRLassoARD(gpflow.models.GPR):
    
    def __init__(self, data, kernel, lasso):
        super(GPRLassoARD, self).__init__(data, kernel)
        self.lasso = lasso
    
    def lml_lasso(self):
        return self.log_marginal_likelihood() - self.lasso*tf.math.reduce_sum(tf.abs(tf.linalg.diag(self.kernel.lengthscales)))
    
    def maximum_log_likelihood_objective(self):
        return self.lml_lasso()

class GPRLassoFullCholesky(gpflow.models.GPR):
    
    def __init__(self, data, kernel, lasso):
        super(GPRLassoFullCholesky, self).__init__(data, kernel)
        self.lasso = lasso
    
    def lml_lasso(self):
        if len(self.lasso) == 2:
            return self.log_marginal_likelihood() - self.lasso[0]*tf.math.reduce_sum(tf.abs(self.kernel.off_diagonal)) - self.lasso[1]*tf.math.reduce_sum(tf.abs(self.kernel.diagonal))
        else:
            paddings = tf.constant([[1, 0,], [0, 1]])
            L_as_matrix = tfp.math.fill_triangular(self.kernel.off_diagonal)
            L_as_matrix = tf.pad(L_as_matrix, paddings, "CONSTANT")
            L = tf.linalg.set_diag(L_as_matrix, self.kernel.diagonal)
            return self.log_marginal_likelihood() - self.lasso*tf.math.reduce_sum(tf.abs(L @ tf.transpose(L)))

    
    def maximum_log_likelihood_objective(self):
        return self.lml_lasso()