functions {
  
  matrix gp_full_cov(vector[] x,
                     real alpha,
                     matrix L,
                     int D) {
    int N = size(x);
    //cov_matrix[N, N] K;
    matrix[N, N] K;
    real sq_alpha = square(alpha);
    for (i in 1:(N-1)) {
      K[i,i] = sq_alpha;
      for (j in (i+1):N) {
        vector[D] L_x = L'*(x[i]-x[j]);
        K[i, j] = sq_alpha * exp(-0.5 * dot_self(L_x));
        K[j, i] = K[i, j];
      }
    }
    K[N, N] = sq_alpha;

    return K;
  }

  matrix to_triangular(vector y_basis, int K) {
    matrix[K, K] y = rep_matrix(0, K, K);    
    int pos = 1;
    for (i in 1:K) {
      for (j in 1:i) {
        y[j, i] = y_basis[pos];
        pos += 1;
      }
    }
    return y;
  }
    
}
data {
  int<lower=1> N;
  int<lower=1> D;
  real D_real_inv;
  vector[D] x[N];
  vector[N] y;
  int<lower=1> N_test;
  vector[D] x_test[N_test];
}
transformed data {
  vector[N] mu = rep_vector(0, N);
  real delta = 1e-9;
}
parameters {
  cholesky_factor_cov[D] transformed_L;
  //vector[D * (D + 1) / 2] vectorized_L;
  real<lower=0> alpha;
  real<lower=0> sigma;
}

transformed parameters {
  //matrix[D, D] transformed_L = to_triangular(vectorized_L, D);
  cov_matrix[D] P = transformed_L * transformed_L';
}


model {
  //cholesky_factor_cov[N] L_K;
  matrix[N, N] L_K;
  {
    //cov_matrix[N, N] K = gp_full_cov(x, alpha, transformed_L, D);
    matrix[N, N] K = gp_full_cov(x, alpha, transformed_L, D);
    real sq_sigma = square(sigma);

    // diagonal elements
    for (n in 1:N)
      K[n, n] = K[n, n] + sq_sigma;
    L_K = cholesky_decompose(K);
  }

  matrix[D,D] S = rep_matrix(0, D, D);
  for (d in 1:D) {
    S[d,d] = 1.0;
  }

  P ~ wishart(D, S);
  y ~ multi_normal_cholesky(mu, L_K);
}