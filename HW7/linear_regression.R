library(ggplot2)

# dataset
set.seed(2)
n <- 100
x <- runif(n, -1, 1)
y <- 2 * x - 1 + rnorm(n)
df       <- data.frame(x, y)
df_small <- df[1:3,]

plot(x, y)
points(df_small$x, df_small$y, col = "red", cex = 2, pch = 16)

# "classic" linear regression
my_lr <- lm(y ~ x, df)
abline(my_lr, col = "blue")
print(summary(my_lr))
my_lr <- lm(y ~ x, df_small)
abline(my_lr, col = "red")

# Bayesian linear regression 
# (low scale = high+ regularization)
library(rstanarm)
library(rstan)
my_blr <- rstanarm::stan_glm(y ~ x, df, 
                             family = gaussian(),
                             chains = 1, iter = 5000,
                             prior = normal(location = 0, scale = NULL, autoscale = F))
samples <- extract(my_blr$stanfit)

plot(df$x, df$y)
points(df_small$x, df_small$y, col = "red", cex = 2, pch = 16)

for (i in 1:1000) {
  abline(a = samples$alpha[i], b = samples$beta[i],
         col = rgb(red = 0, green = 0, blue = 1, alpha = 0.05))
}


# Examples:
# - inference (P(beta > 2 | data))

mean(samples$beta > 2)

# - predictive uncertainty = process noise (sigma) + parameter uncertainty (more robust)
# - regularization
# - entire dataset
