library(xgboost); library(readr); library(stringr); library(car)
library(dplyr);library(tidyr);library(caret);library(outliers);library(lubridate); library(tidyverse);
library(adabag);library(mclust);library(foreach);library(rsample);library(glmnet);library(ade4);
library(onehot);library(reshape2);library(DescTools);library(mlbench) ;library(psych);library(magrittr)

set.seed(1234)

Train <- read.csv("/aaron/Scheduling/Data/Train.csv", header = TRUE)
Validation <- read.csv("/aaron/Scheduling/Data/Validation.csv", header = TRUE)
Test <- read.csv("/aaron/Scheduling/Data/Test.csv", header = TRUE)

Train$Data <- 'Train'
Validation$Data <- 'Validation'
Test$Data <- 'Test'

ALL.data <- rbind(Train,Validation,Test);dim(ALL.data)

ALL.data$Key <- paste0(ALL.data$DATE,"_",ALL.data$PatientNo)

## Procedures
Procedure <- ALL.data %>% select(Key, Proced1, Proced2, Proced3, Proced4, Proced5, Proced6)
ALL.data %<>% select(-Proced1, -Proced2, -Proced3, -Proced4, -Proced5, -Proced6)
colnames(Procedure) <- c('Key','Proced1','Proced2','Proced3','Proced4','Proced5','Proced6')
PCD <- dcast(melt(Procedure ,id.var="Key"), Key ~ value, length) %>% select(-'NA');dim(PCD)

## Diagnosis code (ICD-10 code)
Diagnosis <- ALL.data %>% select(Key, Diag1, Diag2, Diag3, Diag4, Diag5, Diag6)
ALL.data %<>% select(-Diag1, -Diag2, -Diag3, -Diag4, -Diag5, -Diag6)
colnames(Diagnosis) <- c('Key','Diag1','Diag2','Diag3','Diag4','Diag5','Diag6')
DIA <- dcast(melt(Diagnosis ,id.var="Key"), Key ~ value, length) %>% select(-'NA'); dim(DIA)

ALL.data <- list(ALL.data, PCD, DIA) %>% reduce(left_join, by = "Key")
ALL.data %<>% select(-DATE, -PatientNo, -Key); dim(ALL.data)

#######################################################################################################

ALL.data$ASA %<>% as.factor()
ALL.data$Hypertension %<>% as.factor()
ALL.data$DivNo %<>% as.factor()
ALL.data$weekday %<>% as.factor()
ALL.data$TimeofDay %<>% as.factor()
ALL.data$OpType %<>% as.factor()
ALL.data$TeamSize %<>% as.numeric()
ALL.data$SexName %<>% as.factor()
ALL.data$AnaValue %<>% as.factor()
ALL.data$Data %<>% as.factor()
ALL.data$BMI %<>% as.factor()

#########  One-hot encoding  #################
encoder <- onehot(ALL.data,max_levels=10000)
ALL.data_one_hot <- as.data.frame(predict(encoder, ALL.data)); dim(ALL.data_one_hot)
colnames(ALL.data_one_hot) <- gsub('[=]',"_",colnames(ALL.data_one_hot))

################################################

X_train <- ALL.data_one_hot[ALL.data_one_hot$Data_Train ==1,] %>% select(-in_mins,-Data_Train,-Data_Validation,-Data_Test);dim(X_train)
X_validation <- ALL.data_one_hot[ALL.data_one_hot$Data_Validation ==1,] %>% select(-in_mins,-Data_Train,-Data_Validation,-Data_Test);dim(X_validation)
X_test <- ALL.data_one_hot[ALL.data_one_hot$Data_Test ==1 ,] %>% select(-in_mins,-Data_Train,-Data_Validation,-Data_Test);dim(X_test)

Y_train <- ALL.data_one_hot[ALL.data_one_hot$Data_Train ==1,] %>% select(in_mins);dim(Y_train)
Y_validation <- ALL.data_one_hot[ALL.data_one_hot$Data_Validation ==1,] %>% select(in_mins);dim(Y_validation)
Y_test <- ALL.data_one_hot[ALL.data_one_hot$Data_Test ==1 ,] %>% select(in_mins);dim(Y_test)


Names <- colnames(X_train)
X_train %<>% as.matrix()
colnames(X_train) <- Names
Y_train %<>% unlist()

Names <- colnames(X_validation)
X_validation %<>% as.matrix()
colnames(X_validation) <- Names
Y_validation %<>% unlist()

Names <- colnames(X_test)
X_test %<>% as.matrix()
colnames(X_test) <- Names
Y_test %<>% unlist()


XGB = xgb.cv(data = X_train, label = log(Y_train) , nrounds = 1000,
             objective = "reg:squarederror",
             eta = 0.5,
             nfold = 5,
             max_depth = 3,
             subsample = 1,
             colsample_bytree = 1,
             gamma = 0.25,
             alpha = 1)

which.min(XGB$evaluation_log$test_rmse_mean)
Nrounds <- which.min(XGB$evaluation_log$test_rmse_mean)
#Nrounds = 635


# build the model
XGB = xgboost(data = X_train, label = log(Y_train) , nrounds = Nrounds,
              objective = "reg:squarederror",
              eta = 0.5,
              max_depth = 3,
              subsample = 1,
              colsample_bytree = 1,
              gamma = 0.25,
              alpha = 1)

##### training set model evaluation

train.pred <- predict(XGB,newdata=X_train)
train.y    <- log(Y_train)

SS.total      <- sum((train.y - mean(train.y))^2)
SS.residual   <- sum((train.y - train.pred)^2)
SS.regression <- sum((train.pred - mean(train.y))^2)
train.rsq <- 1 - SS.residual/SS.total  
train.rsq

Delta <- sapply(exp(train.pred),function(x) min(max(c(0.15*x, 15)),60))

XGB_dist_train <-
  ifelse(Y_train >= exp(train.pred) + Delta,"overage",
         ifelse(Y_train <= exp(train.pred) - Delta,"underage","within"))
table(XGB_dist_train)/sum(table(XGB_dist_train))

MAE(Y_train , exp(train.pred))

#### validation set evaluation

validation.pred <- predict(XGB,newdata=X_validation)
validation.y    <- log(Y_validation)

SS.total      <- sum((validation.y - mean(validation.y))^2)
SS.residual   <- sum((validation.y - validation.pred)^2)
SS.regression <- sum((validation.pred - mean(validation.y))^2)
validation.rsq <- 1 - SS.residual/SS.total  
validation.rsq

Delta <- sapply(exp(validation.pred),function(x) min(max(c(0.15*x, 15)),60))

XGB_dist_validation <-
  ifelse(Y_validation >= exp(validation.pred) + Delta,"overage",
         ifelse(Y_validation <= exp(validation.pred) - Delta,"underage","within"))
table(XGB_dist_validation)/sum(table(XGB_dist_validation))

MAE(Y_validation , exp(validation.pred))


##### test set evaluation

test.pred <- predict(XGB,newdata=X_test)
test.y    <- log(Y_test)

SS.total      <- sum((test.y - mean(test.y))^2)
SS.residual   <- sum((test.y - test.pred)^2)
SS.regression <- sum((test.pred - mean(test.y))^2)
test.rsq <- 1 - SS.residual/SS.total  
test.rsq

Delta <- sapply(exp(test.pred),function(x) min(max(c(0.15*x, 15)),60))

XGB_dist_test <-
  ifelse(Y_test >= exp(test.pred) + Delta,"overage",
         ifelse(Y_test <= exp(test.pred) - Delta,"underage","within"))
table(XGB_dist_test)/sum(table(XGB_dist_test))

MAE(Y_test , exp(test.pred))

sum(Test$in_mins)
sum(exp(test.pred))
sum(abs(Test$in_mins-exp(test.pred)))
sum(abs(Test$in_mins-exp(test.pred)))/sum(Test$in_mins)


saveRDS(XGB, file = "/home/Aaron/Scheduling/Surgery/XGB.rda")