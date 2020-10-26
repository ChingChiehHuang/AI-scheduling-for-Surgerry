library(xgboost); library(readr); library(stringr); library(caret); library(car)
library(dplyr);library(tidyr);library(randomForest);library(caret);library(e1071)
library(outliers);library(lubridate); library(tidyverse); library(rpart); library(adabag);library(mclust)
library(foreach);library(rsample);library(glmnet);library(ade4);library(onehot);library(reshape2)
library(DescTools);library(keras);library(mlbench) ;library(psych);library(magrittr);library(neuralnet)

set.seed(1234)


X_train <- read.csv('/aaron/Scheduling/Data/X_train.csv', header = TRUE)
X_validation <- read.csv('/aaron/Scheduling/Data/X_validation.csv', header = TRUE)
X_test <- read.csv('/aaron/Scheduling/Data/X_test.csv', header = TRUE)
Y_train <- read.csv('/aaron/Scheduling/Data/Y_train.csv', header = TRUE)
Y_validation <- read.csv('/aaron/Scheduling/Data/Y_validation.csv', header = TRUE)
Y_test <- read.csv('/aaron/Scheduling/Data/Y_test.csv', header = TRUE)


Train <- data.frame(Y_train,X_train) 
Validation <- data.frame(Y_validation,X_validation)
Test <- data.frame(Y_test,X_test)



Reg <- lm(in_mins~.,data=Train)


b <- boxcox(Reg)

lambda <- b$x # lambda values

lik <- b$y # log likelihood values for SSE

bc <- cbind(lambda, lik) # combine lambda and lik

sorted_bc <- bc[order(-lik),] # values are sorted to identify the lambda value for the maximum log likelihood for obtaining minimum SSE

head(sorted_bc, n = 10)
