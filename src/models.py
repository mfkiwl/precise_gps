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
        return self.log_marginal_likelihood() - self.lasso*tf.math.reduce_sum(tf.abs(tf.linalg.diag(self.kernel.ell**(-2))))
    
    def maximum_log_likelihood_objective(self):
        return self.lml_lasso()

class GPRLassoFullCholesky(gpflow.models.GPR):
    
    def __init__(self, data, kernel, lasso):
        super(GPRLassoFullCholesky, self).__init__(data, kernel)
        self.lasso = lasso
    
    def lml_lasso(self):
        L = tfp.math.fill_triangular(self.kernel.L)
        return self.log_marginal_likelihood() - self.lasso*tf.math.reduce_sum(tf.abs(L @ tf.transpose(L)))

    
    def maximum_log_likelihood_objective(self):
        return self.lml_lasso()