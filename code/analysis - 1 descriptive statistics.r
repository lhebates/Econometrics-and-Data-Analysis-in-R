## This script imports data from a csv file, and then create a descriptive statistics table using the stargazer package
## created by Leshui He, 2023

##################################################
# check current working directory
getwd()

# set working directory 
#   can also click: Session --> Set Working Directory --> To Source File Location
setwd("~/Dropbox/Teaching/PioneerAcademics/PA Research/data analysis/code")

##################################################
# import data from csv
yelp_res_data <- read.csv("../data/yelp_restaurants_2021.csv", header = TRUE, sep = ",")

# check size of data
dim(yelp_res_data)

# select first ... columns
yelp_res_data_slim <- yelp_res_data[,1:20]

# show first 6 rows
head(yelp_res_data_slim)


##################################################
# descriptive statistics
summary(yelp_res_data_slim)



##################################################
## export descriptive statistics table
# install package if not already installed (only need to run once in your machine)
install.packages("stargazer")

# Load stargazer package
library(stargazer)

# Create a stargazer table
stargazer(yelp_res_data_slim, type = "text", title="Summary statistics Table", digits=1)

# Export to html file (then copy into word, etc.)
stargazer(yelp_res_data_slim, type = "text", title = "Summary Statistics Table", digits=1, out = "../output/summary_stats_table.html")

