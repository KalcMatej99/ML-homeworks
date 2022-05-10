toy_data <- function(n, seed = NULL) {
  set.seed(seed)
  x <- matrix(rnorm(8 * n), ncol = 8)
  z <- 0.4 * x[,1] - 0.5 * x[,2] + 1.75 * x[,3] - 0.2 * x[,4] + x[,5]
  y <- runif(n) > 1 / (1 + exp(-z))
  return (data.frame(x = x, y = y))
}
log_loss <- function(y, p) {
  -(y * log(p) + (1 - y) * log(1 - p))
}


df_huge <- toy_data(100000, 0)

n <- 50
df_dgp <- toy_data(n, 0)

library(stats)

model_h <- glm(y ~ x.1 + x.2 + x.3+ x.4 + x.5+ x.6 + x.7+ x.8, data=df_dgp, family="binomial")


y <- df_huge$y
X <- df_huge[ , seq(1, 8)]

pred <- predict(model_h, newdata = X)
number_of_rows <- nrow(df_huge)
for(i in 1:number_of_rows){
  y_true <- y[i]
  y_pred <- pred[i]
  X_ <- X[i,]
  
  R <- log_loss(y_true, y_pred)
  print(R)
}