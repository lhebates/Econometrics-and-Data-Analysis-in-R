## This script simulates data and estimates a simple OLS model; relies on tidyverse package
## created by Leshui He, 2023

##################################################
# check current working directory
getwd()

# set working directory 
#   can also click: Session --> Set Working Directory --> To Source File Location
setwd("~/Dropbox/Teaching/PioneerAcademics/PA Research/data analysis/code")


##################################################
# install package tidyverse; 
#   if encounter errors, google the error message and check solutions online
#   on mac, one may needs to open terminal and install xcode, see: https://apple.stackexchange.com/questions/254380/why-am-i-getting-an-invalid-active-developer-path-when-attempting-to-use-git-a
install.packages("tidyverse")

# load package
library(tidyverse)

# set random seed for reproducibility
set.seed(123)

# set sample size
n <- 10000

#
tb <- tibble(
  x = rnorm(n),
  u = rnorm(n),
  y = 5.5*x + 12*u
) 

# create scatter plot
plot(x = df$x, y = df$y, xlab = "x", ylab = "y", main = "Scatter Plot of y vs. x")

## estimate regression model
reg_tb <- tb %>% 
  lm(y ~ x, .) %>%
  print()

reg_tb$coefficients

reg_tb$coefficients[1]

tb <- tb %>% 
  mutate(
    yhat1 = predict(lm(y ~ x, .)),
    yhat2 = reg_tb$coefficients[1] + reg_tb$coefficients[2]*x, 
    uhat1 = residuals(lm(y ~ x, .)),
    uhat2 = y - yhat2
  )

summary(tb[-1:-3])

tb %>% 
  lm(y ~ x, .) %>% 
  ggplot(aes(x=x, y=y)) + 
  ggtitle("OLS Regression Line") +
  geom_point(size = 0.05, color = "black", alpha = 0.5) +
  geom_smooth(method = lm, color = "black") +
  annotate("text", x = -1.5, y = 5, color = "red", 
           label = paste("Intercept = ", reg_tb$coefficients[1])) +
  annotate("text", x = 1.5, y = -5, color = "blue", 
           label = paste("Slope =", reg_tb$coefficients[2]))