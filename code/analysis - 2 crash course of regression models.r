## This script simulates data and estimates a simple OLS model
## created by Leshui He, 2023

##################################################
# check current working directory
getwd()

# set working directory 
#   can also click: Session --> Set Working Directory --> To Source File Location
setwd("~/Dropbox/Teaching/PioneerAcademics/PA Research/data analysis/code")

# Load data
# data(mtcars)

##################################################

# set random seed for reproducibility
set.seed(123)

# set sample size
n <- 1000
slope <- 3
intercept <- 1.5

# Create data frame
df <- data.frame(x = rnorm(n), u = rnorm(n))
df$y <- intercept + slope * df$x + df$u

# check dimensions of dataframe
dim(df)

# create scatter plot
plot(x = df$x, y = df$y, xlab = "x", ylab = "y", main = "Scatter Plot of y vs. x")

# regress y on x
#  specifies the formula: y = intercept + slope * x + u
#   to estimate the slope and intercept from the observed data in df
reg_df <- lm(y ~ x, df)
print(reg_df)

# export regression table to html file (then copy into word, etc.)
library(stargazer)
stargazer(reg_df, type = "text", title="OLS Regression Results", digits=4, out = "../output/regression_table_simulation_OLS.html")


##################################################
### prediction using regression model
# predict yhat from x
df$yhat1 <- predict(reg_df)
# predict yhat from x using the estimated slope and intercept
df$yhat2 <- reg_df$coefficients[1] + reg_df$coefficients[2] * df$x
# calculate residuals
df$uhat1 <- residuals(reg_df)
# calculate residuals using the estimated slope and intercept
df$uhat2 <- df$y - df$yhat2

## compare predictions and residuals from the two methods in summary
summary(df[-1:-3])


##################################################
### visualize regression
install.packages("ggplot2")

library(ggplot2)

# create scatter plot with regression line and confidence bands
ggplot(lm(y ~ x, df), aes(x=x, y=y)) + 
ggtitle("OLS Regression Line") +
geom_point(size = 0.05, color = "black", alpha = 0.5) +
geom_smooth(method = lm, color = "black") +
annotate("text", x = -1.5, y = 5, color = "red", 
          label = paste("Intercept = ", reg_df$coefficients[1])) +
annotate("text", x = 1.5, y = -5, color = "blue", 
          label = paste("Slope =", reg_df$coefficients[2]))