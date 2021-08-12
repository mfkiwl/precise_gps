functions {
  
  matrix L_cov_exp_quad_ARD(vector[] x1,
                            vector[] x2,
                            real alpha,
                            vector rho,
                            real delta) {
    int N1 = size(x1);
    int N2 = size(x2);
    matrix[N1, N2] K;
    real sq_alpha = square(alpha);
    vector[size(rho)] squared_rho = rho .* rho;
    for (i in 1:N1) {
      for (j in 1:N2) {
        K[i, j] = sq_alpha
                      * exp(-0.5 * dot_self((x1[i] - x2[j]) .* squared_rho));
      }
    }
    return K;
  }


  vector gp_pred_rng(vector[] x2,
                     vector y1,
                     vector[] x1,
                     real alpha,
                     vector rho,
                     real sigma,
                     real delta) {
    int N1 = rows(y1);
    int N2 = size(x2);
    vector[N2] f2;
    {
      matrix[N1, N1] L_K;
      vector[N1] K_div_y1;
      matrix[N1, N2] k_x1_x2;
      matrix[N1, N2] v_pred;
      vector[N2] f2_mu;
      matrix[N2, N2] cov_f2;
      matrix[N2, N2] diag_delta;
      matrix[N1, N1] K;
      K = L_cov_exp_quad_ARD(x1,x1, alpha, rho, delta);
      for (n in 1:N1)
        K[n, n] = K[n,n] + square(sigma);
      L_K = cholesky_decompose(K);
      K_div_y1 = mdivide_left_tri_low(L_K, y1);
      K_div_y1 = mdivide_right_tri_low(K_div_y1', L_K)';
      k_x1_x2 = L_cov_exp_quad_ARD(x1,x2, alpha, rho, delta);
      f2_mu = (k_x1_x2' * K_div_y1);
      v_pred = mdivide_left_tri_low(L_K, k_x1_x2);
      cov_f2 = L_cov_exp_quad_ARD(x2,x2, alpha, rho, delta) - v_pred' * v_pred;
      diag_delta = diag_matrix(rep_vector(delta, N2));

      f2 = multi_normal_rng(f2_mu, cov_f2 + diag_delta);
    }
    return f2;
  }
    
}
data {
  int<lower=1> N;
  int<lower=1> D;
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
  vector<lower=0>[D] precision;
  real<lower=0> alpha;
  real<lower=0> sigma;
}


model {
  matrix[N, N] L_K;
  {
    matrix[N, N] K = L_cov_exp_quad_ARD(x,x, alpha, precision, delta);
    real sq_sigma = square(sigma);

    // diagonal elements
    for (n1 in 1:N)
      K[n1, n1] = K[n1, n1] + sq_sigma;

    L_K = cholesky_decompose(K);
  }
  
  precision ~ inv_gamma(5, 5);
  alpha ~ std_normal();
  sigma ~ std_normal();

  y ~ multi_normal_cholesky(mu, L_K);
}
generated quantities {
  vector[N_test] f_pred;
  vector[N_test] y_pred;

  f_pred = gp_pred_rng(x_test, y, x, alpha, precision, sigma, delta);
  for (n2 in 1:N_test)
    y_pred[n2] = normal_rng(f_pred[n2], sigma);
}