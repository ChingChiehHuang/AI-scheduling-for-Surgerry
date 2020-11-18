library(xgboost); library(readr); library(stringr); library(car)
library(dplyr);library(tidyr);library(randomForest);library(caret);library(e1071)
library(outliers);library(lubridate); library(tidyverse); library(rpart); library(adabag);library(mclust)
library(foreach);library(rsample);library(glmnet);library(ade4);library(onehot);library(reshape2)
library(DescTools);library(mlbench) ;library(psych);library(magrittr)

ICD <- read.csv("/aaron/Scheduling/Data/ICD_code.csv", header = TRUE)
colnames(ICD) <- c('ICD_9','ICD_10')
ICD$ICD_10 %<>% gsub("\\.[a-z0-9A-Z]+$","",.)
ICD$ICD_9  %<>% gsub("\\.[0-9]+$","",.)
ICD <- ICD[!duplicated(ICD), ]

Data2017 <- read.csv("/aaron/Scheduling/Data/OR_data_106.csv", header = TRUE)
Data2018 <- read.csv("/aaron/Scheduling/Data/OR_data_107.csv", header = TRUE)
Data2019_Jan_June <- read.csv("/aaron/Scheduling/Data/OR_data_Jan-June108.csv", header = TRUE)
Data2019_July_Oct <- read.csv("/aaron/Scheduling/Data/OR_data_July-Oct108.csv", header = TRUE)
Data2019_Nov_Dec <- read.csv("/aaron/Scheduling/Data/OR_data_Nov-Dec108.csv", header = TRUE)
Data2020 <- read.csv("/aaron/Scheduling/Data/OR_data_March-April109.csv", header = TRUE)

Data2018$ASA[Data2018$ASA %in% c('E','I')] <- NA
Data2018$ASA %<>% as.integer()

Data2019 <- bind_rows(Data2019_Jan_June,Data2019_July_Oct,Data2019_Nov_Dec)
Data2017$Proced6 %<>% as.factor()
Data2018$Proced6 %<>% as.factor()
Data2019$Proced6 %<>% as.factor()
Data2020$Proced6 %<>% as.factor()
Data2017$Height %<>% as.numeric()
Data2018$Height %<>% as.numeric()
Data2019$Height %<>% as.numeric()
Data2020$Height %<>% as.numeric()

Train <- bind_rows(Data2017,Data2018,Data2019) %>% as.data.frame();dim(Train)
Test <- Data2020 %>% as.data.frame();dim(Test)


#delete duplicated cases
train_two_proc <- paste0(Train$DATE,'_',Train$PatientNo)
test_two_proc <- paste0(Test$DATE,'_',Test$PatientNo)
Train <- Train[!(duplicated(train_two_proc) | duplicated(train_two_proc, fromLast = TRUE)), ]
Test <- Test[!(duplicated(test_two_proc) | duplicated(test_two_proc, fromLast = TRUE)), ]

Train <- apply(Train,2,function(x)gsub('\\s+', '',x))  %>% as.data.frame()
Test <- apply(Test,2,function(x)gsub('\\s+', '',x))  %>% as.data.frame()

Train %<>% mutate_all(as.character)
Test %<>% mutate_all(as.character)

Train[Train == ''] <- NA
Test[Test == ''] <- NA

Train$age %<>% as.numeric()
Train$Dr_age %<>% as.numeric()
Train$Height %<>% as.numeric()
Train$Weight %<>% as.numeric()

Test$age %<>% as.numeric()
Test$Dr_age %<>% as.numeric()
Test$Height %<>% as.numeric()
Test$Weight %<>% as.numeric()

sapply(Train,class)
sapply(Test,class)

# Chaging rare AnaValue to Others
Train$AnaValue[Train$AnaValue %in% c('BL','DC','PL','','SE') ] <- 'Others'
Test$AnaValue[ !(Test$AnaValue %in% Train$AnaValue) ] <- 'Others'

#transform ICD-9 to ICD-10
Train$Diag1 %<>% gsub("\\.[a-z0-9A-Z]+$","",.); Test$Diag1 %<>% gsub("\\.[a-z0-9A-Z]+$","",.)
Train$Diag2 %<>% gsub("\\.[a-z0-9A-Z]+$","",.); Test$Diag2 %<>% gsub("\\.[a-z0-9A-Z]+$","",.)
Train$Diag3 %<>% gsub("\\.[a-z0-9A-Z]+$","",.); Test$Diag3 %<>% gsub("\\.[a-z0-9A-Z]+$","",.)
Train$Diag4 %<>% gsub("\\.[a-z0-9A-Z]+$","",.); Test$Diag4 %<>% gsub("\\.[a-z0-9A-Z]+$","",.)
Train$Diag5 %<>% gsub("\\.[a-z0-9A-Z]+$","",.); Test$Diag5 %<>% gsub("\\.[a-z0-9A-Z]+$","",.)
Train$Diag6 %<>% gsub("\\.[a-z0-9A-Z]+$","",.); Test$Diag6 %<>% gsub("\\.[a-z0-9A-Z]+$","",.)

Train$Diag1[Train$Diag1 %in% ICD$ICD_9] <- ICD$ICD_10[match(Train$Diag1[Train$Diag1 %in% ICD$ICD_9],ICD$ICD_9)]
Train$Diag2[Train$Diag2 %in% ICD$ICD_9] <- ICD$ICD_10[match(Train$Diag2[Train$Diag2 %in% ICD$ICD_9],ICD$ICD_9)]
Train$Diag3[Train$Diag3 %in% ICD$ICD_9] <- ICD$ICD_10[match(Train$Diag3[Train$Diag3 %in% ICD$ICD_9],ICD$ICD_9)]
Train$Diag4[Train$Diag4 %in% ICD$ICD_9] <- ICD$ICD_10[match(Train$Diag4[Train$Diag4 %in% ICD$ICD_9],ICD$ICD_9)]
Train$Diag5[Train$Diag5 %in% ICD$ICD_9] <- ICD$ICD_10[match(Train$Diag5[Train$Diag5 %in% ICD$ICD_9],ICD$ICD_9)]
Train$Diag6[Train$Diag6 %in% ICD$ICD_9] <- ICD$ICD_10[match(Train$Diag6[Train$Diag6 %in% ICD$ICD_9],ICD$ICD_9)]

Test$Diag1[Test$Diag1 %in% ICD$ICD_9] <- ICD$ICD_10[match(Test$Diag1[Test$Diag1 %in% ICD$ICD_9],ICD$ICD_9)]
Test$Diag2[Test$Diag2 %in% ICD$ICD_9] <- ICD$ICD_10[match(Test$Diag2[Test$Diag2 %in% ICD$ICD_9],ICD$ICD_9)]
Test$Diag3[Test$Diag3 %in% ICD$ICD_9] <- ICD$ICD_10[match(Test$Diag3[Test$Diag3 %in% ICD$ICD_9],ICD$ICD_9)]
Test$Diag4[Test$Diag4 %in% ICD$ICD_9] <- ICD$ICD_10[match(Test$Diag4[Test$Diag4 %in% ICD$ICD_9],ICD$ICD_9)]
Test$Diag5[Test$Diag5 %in% ICD$ICD_9] <- ICD$ICD_10[match(Test$Diag5[Test$Diag5 %in% ICD$ICD_9],ICD$ICD_9)]
Test$Diag6[Test$Diag6 %in% ICD$ICD_9] <- ICD$ICD_10[match(Test$Diag6[Test$Diag6 %in% ICD$ICD_9],ICD$ICD_9)]

#day of week
Sys.setlocale("LC_TIME", "English")
Train$DATE <- paste0(as.numeric(substr(Train$DATE,1,3))+1911, "-",
                     substr(Train$DATE,4,5), "-", substr(Train$DATE,6,7))
Test$DATE <- paste0(as.numeric(substr(Test$DATE,1,3))+1911, "-",
                    substr(Test$DATE,4,5), "-", substr(Test$DATE,6,7))
Train$weekday <- weekdays(as.Date(Train$DATE,format='%Y-%m-%d'))
Test$weekday <- weekdays(as.Date(Test$DATE,format='%Y-%m-%d'))

Train <- Train[!is.na(Train$In) & nchar(Train$In)>2,]; Test <- Test[!is.na(Test$In) & nchar(Test$In)>2,]
Train <- Train[!is.na(Train$OUT) & nchar(Train$OUT)>2,]; Test <- Test[!is.na(Test$OUT) & nchar(Test$OUT)>2,]


Train$OUT <-
  sapply(Train$OUT,function(x){
    if(nchar(x) == 3){
      return( paste0(substr(x,1,1), ":", substr(x,2,3))  )
    }else if(nchar(x) == 4){
      return( paste0(substr(x,1,2), ":", substr(x,3,4))  )
    }else{
      return(NA)
    }})
Train$In <- 
  sapply(Train$In,function(x){
    if(nchar(x) == 3){
      return( paste0(substr(x,1,1), ":", substr(x,2,3))  )
    }else if(nchar(x) == 4){
      return( paste0(substr(x,1,2), ":", substr(x,3,4))  )
    }else{
      return(NA)
    }})
Test$OUT <-
  sapply(Test$OUT,function(x){
    if(nchar(x) == 3){
      return( paste0(substr(x,1,1), ":", substr(x,2,3))  )
    }else if(nchar(x) == 4){
      return( paste0(substr(x,1,2), ":", substr(x,3,4))  )
    }else{
      return(NA)
    }})
Test$In <- 
  sapply(Test$In,function(x){
    if(nchar(x) == 3){
      return( paste0(substr(x,1,1), ":", substr(x,2,3))  )
    }else if(nchar(x) == 4){
      return( paste0(substr(x,1,2), ":", substr(x,3,4))  )
    }else{
      return(NA)
    }})


Train$exe_times <-  as.POSIXct(paste0(Train$DATE," ",Train$In))
Test$exe_times <-  as.POSIXct(paste0(Test$DATE," ",Test$In))


#delete Optype = 21,31
Train <- Train[Train$OpType %in% c('1','11'),];dim(Train)
Test <- Test[Test$OpType %in% c('1','11'),];dim(Test)

#Delete patients age <20
Train <- Train[Train$age >= 20,];dim(Train)
Test <- Test[Test$age >= 20,];dim(Test)

#Delete pregnant patients
Train <- Train[!grepl("^81",Train$Proced1),];dim(Train)
Test <- Test[!grepl("^81",Test$Proced1),];dim(Test)

#in_mins 
Train$OUT %<>% hm()
Train$In %<>% hm()
Test$OUT %<>% hm()
Test$In %<>% hm()

Train$in_mins <- as.numeric(Train$OUT -Train$In)/60
Train <- Train[!is.na(Train$in_mins),];dim(Train)
Train$in_mins[Train$in_mins < 0] <- 1440 + Train$in_mins[Train$in_mins < 0]

Test$in_mins <- as.numeric(Test$OUT -Test$In)/60
Test <- Test[!is.na(Test$in_mins),];dim(Test)
Test$in_mins[Test$in_mins < 0] <- 1440 + Test$in_mins[Test$in_mins < 0]

### The number of previous surgeries and total surgical durations
r1 <- 
  foreach(i = 1:dim(Train)[1], .combine=rbind) %dopar% {
    Dr_time <- Train$exe_times[i]
    Dr_group <- Train[Train$DrID %in% Train$DrID[i],]
    cbind(
      sum(Dr_group$exe_times >= Dr_time-days(1) & Dr_group$exe_times <= Dr_time-1 ,na.rm = TRUE),
      sum(Dr_group$in_mins[  Dr_group$exe_times >= Dr_time-days(1) & Dr_group$exe_times <= Dr_time-1] ,na.rm = TRUE),
      sum(Dr_group$exe_times >= Dr_time-days(7) & Dr_group$exe_times <= Dr_time-1 ,na.rm = TRUE),
      sum(Dr_group$in_mins[  Dr_group$exe_times >= Dr_time-days(7) & Dr_group$exe_times <= Dr_time-1] ,na.rm = TRUE)
    )}

Train$Opcount_1d <- r1[,1]
Train$Optoltime_1d <- r1[,2]
Train$Opcount_7d <- r1[,3]
Train$Optoltime_7d <- r1[,4]

r2 <- 
  foreach(i = 1:dim(Test)[1], .combine=rbind) %dopar% {
    Dr_time <- Test$exe_times[i]
    Dr_group <- Test[Test$DrID %in% Test$DrID[i],]
    cbind(
      sum(Dr_group$exe_times >= Dr_time-days(1) & Dr_group$exe_times <= Dr_time-1 ,na.rm = TRUE),
      sum(Dr_group$in_mins[  Dr_group$exe_times >= Dr_time-days(1) & Dr_group$exe_times <= Dr_time-1] ,na.rm = TRUE),
      sum(Dr_group$exe_times >= Dr_time-days(7) & Dr_group$exe_times <= Dr_time-1 ,na.rm = TRUE),
      sum(Dr_group$in_mins[  Dr_group$exe_times >= Dr_time-days(7) & Dr_group$exe_times <= Dr_time-1] ,na.rm = TRUE)
    )}

Test$Opcount_1d <- r2[,1]
Test$Optoltime_1d <- r2[,2]
Test$Opcount_7d <- r2[,3]
Test$Optoltime_7d <- r2[,4]

#time of day
breaks <- hour(hm("00:00", "6:00", "12:00", "18:00", "23:59"))
labels <- c("Night", "Morning", "Afternoon", "Evening")
Train$TimeofDay <- cut(x=hour(Train$In), breaks = breaks, labels = labels, include.lowest=TRUE)
Test$TimeofDay <- cut(x=hour(Test$In), breaks = breaks, labels = labels, include.lowest=TRUE)

#delete procedure duration > 10 hrs or < 10 mins
Train <- Train[(Train$in_mins < 60*10 & Train$in_mins >10),];dim(Train)
Test <- Test[(Test$in_mins < 60*10 & Test$in_mins >10),];dim(Test)

#Doctors
Train$DrType2[!grepl("^[dD][0-9]+",Train$DrType2)] <- NA
Train$DrType3[!grepl("^[dD][0-9]+",Train$DrType3)] <- NA
Train$DrType4[!grepl("^[dD][0-9]+",Train$DrType4)] <- NA
Train$DrType5[!grepl("^[dD][0-9]+",Train$DrType5)] <- NA

Test$DrType2[!grepl("^[dD][0-9]+",Test$DrType2)] <- NA
Test$DrType3[!grepl("^[dD][0-9]+",Test$DrType3)] <- NA
Test$DrType4[!grepl("^[dD][0-9]+",Test$DrType4)] <- NA
Test$DrType5[!grepl("^[dD][0-9]+",Test$DrType5)] <- NA

#doctor teamSize
Train$TeamSize <- rowSums(!is.na(cbind(Train$DrID, Train$DrType2, Train$DrType3, Train$DrType4, Train$DrType5)))
Test$TeamSize <- rowSums(!is.na(cbind(Test$DrID, Test$DrType2, Test$DrType3, Test$DrType4, Test$DrType5)))


#ICD-10
Train$Diag1[!grepl("^[A-Za-z]",Train$Diag1)] <- NA
Train$Diag2[!grepl("^[A-Za-z]",Train$Diag2)] <- NA
Train$Diag3[!grepl("^[A-Za-z]",Train$Diag3)] <- NA
Train$Diag4[!grepl("^[A-Za-z]",Train$Diag4)] <- NA
Train$Diag5[!grepl("^[A-Za-z]",Train$Diag5)] <- NA

Test$Diag1[!grepl("^[A-Za-z]",Test$Diag1)] <- NA
Test$Diag2[!grepl("^[A-Za-z]",Test$Diag2)] <- NA
Test$Diag3[!grepl("^[A-Za-z]",Test$Diag3)] <- NA
Test$Diag4[!grepl("^[A-Za-z]",Test$Diag4)] <- NA
Test$Diag5[!grepl("^[A-Za-z]",Test$Diag5)] <- NA

#ASA
Train$ASA[Train$ASA >7] <- 8
Test$ASA[Test$ASA >7] <- 8

# Procedure
Train$Proced1[Train$Proced1 == 'XXX'] <- NA; Test$Proced1[Test$Proced1 == 'XXX'] <- NA
Train$Proced2[Train$Proced2 == 'XXX'] <- NA; Test$Proced2[Test$Proced2 == 'XXX'] <- NA
Train$Proced3[Train$Proced3 == 'XXX'] <- NA; Test$Proced3[Test$Proced3 == 'XXX'] <- NA
Train$Proced4[Train$Proced4 == 'XXX'] <- NA; Test$Proced4[Test$Proced4 == 'XXX'] <- NA
Train$Proced5[Train$Proced5 == 'XXX'] <- NA; Test$Proced5[Test$Proced5 == 'XXX'] <- NA
Train$Proced6[Train$Proced6 == 'XXX'] <- NA; Test$Proced6[Test$Proced6 == 'XXX'] <- NA

# Delete surgeries were all missingness
Train <- Train[ rowSums(!is.na(data.frame(Train$Proced1, Train$Proced2, Train$Proced3, Train$Proced4, Train$Proced5, Train$Proced6))) != 0,]
Test <- Test[ rowSums(!is.na(data.frame(Test$Proced1, Test$Proced2, Test$Proced3, Test$Proced4, Test$Proced5, Test$Proced6))) != 0,]
dim(Train); dim(Test)

#number of surgeries
Train$Nproced <- rowSums(!is.na(cbind(Train$Proced1, Train$Proced2, Train$Proced3, Train$Proced4, Train$Proced5, Train$Proced6)))
Test$Nproced <- rowSums(!is.na(cbind(Test$Proced1, Test$Proced2, Test$Proced3, Test$Proced4, Test$Proced5, Test$Proced6)))

# Delete diagnosis codes were all missingness
Train <- Train[ rowSums(!is.na(data.frame(Train$Diag1, Train$Diag2, Train$Diag3, Train$Diag4, Train$Diag5, Train$Diag6))) != 0,]
Test <- Test[ rowSums(!is.na(data.frame(Test$Diag1, Test$Diag2, Test$Diag3, Test$Diag4, Test$Diag5, Test$Diag6))) != 0,]
dim(Train); dim(Test)


Key <- paste0(Train$DATE,"_",Train$PatientNo)

## Procedures
Procedure <- data.frame(Key, Train$Proced1, Train$Proced2, Train$Proced3, Train$Proced4, Train$Proced5, Train$Proced6)
colnames(Procedure) <- c('Key','Proced1','Proced2','Proced3','Proced4','Proced5','Proced6')
PCD <- melt(Procedure ,id.var="Key"); dim(PCD)
PCD_name <- names(table(PCD$value))[table(PCD$value) > 20]
PCD_name


Train$Proced1[!(Train$Proced1 %in% PCD_name) & !is.na(Train$Proced1)] <- 'Other_proced'
Train$Proced2[!(Train$Proced2 %in% PCD_name) & !is.na(Train$Proced2)] <- 'Other_proced'
Train$Proced3[!(Train$Proced3 %in% PCD_name) & !is.na(Train$Proced3)] <- 'Other_proced'
Train$Proced4[!(Train$Proced4 %in% PCD_name) & !is.na(Train$Proced4)] <- 'Other_proced'
Train$Proced5[!(Train$Proced5 %in% PCD_name) & !is.na(Train$Proced5)] <- 'Other_proced'
Train$Proced6[!(Train$Proced6 %in% PCD_name) & !is.na(Train$Proced6)] <- 'Other_proced'

Test$Proced1[!(Test$Proced1 %in% PCD_name) & !is.na(Test$Proced1)] <- 'Other_proced'
Test$Proced2[!(Test$Proced2 %in% PCD_name) & !is.na(Test$Proced2)] <- 'Other_proced'
Test$Proced3[!(Test$Proced3 %in% PCD_name) & !is.na(Test$Proced3)] <- 'Other_proced'
Test$Proced4[!(Test$Proced4 %in% PCD_name) & !is.na(Test$Proced4)] <- 'Other_proced'
Test$Proced5[!(Test$Proced5 %in% PCD_name) & !is.na(Test$Proced5)] <- 'Other_proced'
Test$Proced6[!(Test$Proced6 %in% PCD_name) & !is.na(Test$Proced6)] <- 'Other_proced'

## Diagnosis code (ICD-10 code)

Diagnosis <- data.frame(Key, Train$Diag1, Train$Diag2, Train$Diag3, Train$Diag4, Train$Diag5, Train$Diag6)
colnames(Diagnosis) <- c('Key','Diag1','Diag2','Diag3','Diag4','Diag5','Diag6')

DIA <- melt(Diagnosis ,id.var="Key"); dim(DIA)
DIA_name <- names(table(DIA$value))[table(DIA$value) > 20]
DIA_name

# categorized the diagnosis code

Train$Diag1[!(Train$Diag1 %in% DIA_name) & !is.na(Train$Diag1)] <- 'Other_diag'
Train$Diag2[!(Train$Diag2 %in% DIA_name) & !is.na(Train$Diag2)] <- 'Other_diag'
Train$Diag3[!(Train$Diag3 %in% DIA_name) & !is.na(Train$Diag3)] <- 'Other_diag'
Train$Diag4[!(Train$Diag4 %in% DIA_name) & !is.na(Train$Diag4)] <- 'Other_diag'
Train$Diag5[!(Train$Diag5 %in% DIA_name) & !is.na(Train$Diag5)] <- 'Other_diag'
Train$Diag6[!(Train$Diag6 %in% DIA_name) & !is.na(Train$Diag6)] <- 'Other_diag'

Test$Diag1[!(Test$Diag1 %in% DIA_name) & !is.na(Test$Diag1)] <- 'Other_diag'
Test$Diag2[!(Test$Diag2 %in% DIA_name) & !is.na(Test$Diag2)] <- 'Other_diag'
Test$Diag3[!(Test$Diag3 %in% DIA_name) & !is.na(Test$Diag3)] <- 'Other_diag'
Test$Diag4[!(Test$Diag4 %in% DIA_name) & !is.na(Test$Diag4)] <- 'Other_diag'
Test$Diag5[!(Test$Diag5 %in% DIA_name) & !is.na(Test$Diag5)] <- 'Other_diag'
Test$Diag6[!(Test$Diag6 %in% DIA_name) & !is.na(Test$Diag6)] <- 'Other_diag'

#Unusual DivNo (< 20) categoried as 'others'
Train$DivNo[Train$DivNo %in%  names(table(Train$DivNo))[table(Train$DivNo) < 20]] <- 'Others'
Test$DivNo[!(Test$DivNo %in% Train$DivNo)] <- 'Others'

Train$ASA[is.na(Train$ASA)] <- 'unknown'
Train$Hypertension[is.na(Train$Hypertension)] <- 'unknown'

Test$ASA[is.na(Test$ASA)] <- 'unknown'
Test$Hypertension[is.na(Test$Hypertension)] <- 'unknown'

## BMI
Train$BMI <- Train$Weight/((Train$Height/100)^2)
Train$BMI <- 
  ifelse(Train$BMI < 18.5,'below',
         ifelse(Train$BMI < 23,'within',
                ifelse(Train$BMI < 27,'above',
                       'Over_above')))
Train$BMI[is.na(Train$BMI)] <- 'unknown'


Test$BMI <- Test$Weight/((Test$Height/100)^2)
Test$BMI <- 
  ifelse(Test$BMI < 18.5,'below',
         ifelse(Test$BMI < 23,'within',
                ifelse(Test$BMI < 27,'above',
                       'Over_above')))
Test$BMI[is.na(Test$BMI)] <- 'unknown'


Train$ASA %<>% as.factor()
Train$Hypertension %<>% as.factor()
Train$DivNo %<>% as.factor()
Train$weekday %<>% as.factor()
Train$TimeofDay %<>% as.factor()
Train$OpType %<>% as.factor()
Train$TeamSize %<>% as.factor()
Train$SexName %<>% as.factor()
Train$AnaValue %<>% as.factor()
Train$Diag1 %<>% as.factor()
Train$Diag2 %<>% as.factor()
Train$Diag3 %<>% as.factor()
Train$Diag4 %<>% as.factor()
Train$Diag5 %<>% as.factor()
Train$Diag6 %<>% as.factor()
Train$Proced1 %<>% as.factor()
Train$Proced2 %<>% as.factor()
Train$Proced3 %<>% as.factor()
Train$Proced4 %<>% as.factor()
Train$Proced5 %<>% as.factor()
Train$Proced6 %<>% as.factor()
Train$BMI %<>% as.factor()


Test$ASA %<>% as.factor()
Test$Hypertension %<>% as.factor()
Test$DivNo %<>% as.factor()
Test$weekday %<>% as.factor()
Test$TimeofDay %<>% as.factor()
Test$OpType %<>% as.factor()
Test$TeamSize %<>% as.factor()
Test$SexName %<>% as.factor()
Test$AnaValue %<>% as.factor()
Test$Diag1 %<>% as.factor()
Test$Diag2 %<>% as.factor()
Test$Diag3 %<>% as.factor()
Test$Diag4 %<>% as.factor()
Test$Diag5 %<>% as.factor()
Test$Diag6 %<>% as.factor()
Test$Proced1 %<>% as.factor()
Test$Proced2 %<>% as.factor()
Test$Proced3 %<>% as.factor()
Test$Proced4 %<>% as.factor()
Test$Proced5 %<>% as.factor()
Test$Proced6 %<>% as.factor()
Test$BMI %<>% as.factor()

#########################################################

Train %<>% select(-Weight,-Height,-In,-OUT,-exe_times)
Test %<>% select(-Weight,-Height,-In,-OUT,-exe_times)

# Data spliting
set.seed(12345)
ind <- sample(c(0,1),size=dim(Train)[1],prob=c(.8, .2) ,replace = T)

Validation <- Train[ind==1,];dim(Validation)
Train <- Train[ind==0,];dim(Train)

write.csv(Train,file='/aaron/Scheduling/Data/Train.csv',row.names = FALSE)
write.csv(Validation,file='/aaron/Scheduling/Data/Validation.csv',row.names = FALSE)
write.csv(Test,file='/aaron/Scheduling/Data/Test.csv',row.names = FALSE)
