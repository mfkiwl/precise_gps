library(clusterGeneration)

matrix <- genPositiveDefMat(8)
samples <- MASS::mvrnorm(n = 2000, mu = c(0,0,8,-2,3.5,10,2,-1), matrix$Sigma, tol = 1e-06, empirical = FALSE)

generate_y <- function(samples) {
  f_true <- c()
  shape <- dim(samples)
  len <- shape[1]
  for (sample in 1:len) {
    s <- samples[sample,]
    f_true <- append(f_true, s[1] + 2*s[2] - 0.5*s[3]*s[3] + s[1]*s[5] + s[4]*s[2] + s[5]*s[4])
  }
  df <- data.frame(f_true)
  colnames(df) <- c("f_true")
  dimensions <- shape[2]
  for (d in 1:dimensions) {
    df[,ncol(df) + 1] <- samples[,d]
    colnames(df)[ncol(df)] <- paste0("x", d) 
  }
  
  return(df)
}
out <- generate_y(samples)
write.csv(matrix, paste(getwd(), "/data/matrix.csv", sep =""))
write.csv(out, paste(getwd(), "/data/2000data.csv", sep =""))
