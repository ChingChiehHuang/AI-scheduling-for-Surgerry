library(xgboost); library(readr); library(stringr); library(caret); library(car)
library(dplyr);library(tidyr);library(randomForest);library(caret);library(e1071)
library(outliers);library(lubridate); library(tidyverse); library(rpart); library(adabag);library(mclust)
library(foreach);library(rsample);library(glmnet);library(ade4);library(onehot);library(reshape2)
library(DescTools);library(keras);library(mlbench) ;library(psych);library(magrittr);library(neuralnet)

##################################################################################
############################ Variable Importance #################################
##################################################################################

imp <- readr::read_csv('D:/AI Scheduling/AI-scheduling-for-Surgerry/Data/XGB_importance.csv',col_names=T)
Train <- read.csv("D:/AI Scheduling/AI-scheduling-for-Surgerry/Data/Train.csv", header = TRUE)

Procedure <- 
  data.frame(Train$Proced1,Train$Proced2,Train$Proced3,
             Train$Proced4,Train$Proced5,Train$Proced6) %>% unlist() %>% unique() %>% as.character()
Surgeon <- 
  data.frame(Train$DrID,Train$DrType2,Train$DrType3,
             Train$DrType4,Train$DrType5) %>% unlist() %>% unique() %>% as.character()
Diag <- 
  data.frame(Train$Diag1,Train$Diag2,Train$Diag3,
             Train$Diag4,Train$Diag5,Train$Diag6) %>% unlist() %>% unique() %>% as.character()

imp$Feature %<>% gsub("^DivNo_[a-zA-Z0-9:punct:]+","DivNo",.)
imp$Feature %<>% gsub("^AnaValue_[a-zA-Z0-9:punct:]+","AnaValue",.)
imp$Feature %<>% gsub("^Hypertension_[a-zA-Z0-9:punct:]+","Hypertension",.)
imp$Feature %<>% gsub("^OpType_[a-zA-Z0-9:punct:]+","OpType",.)
imp$Feature %<>% gsub("^OpRoom_[a-zA-Z0-9:punct:]+","OpRoom",.)
imp$Feature %<>% gsub("^TeamSize_[a-zA-Z0-9:punct:]+","TeamSize",.)
imp$Feature %<>% gsub("^TimeofDay_[a-zA-Z0-9:punct:]+","TimeofDay",.)
imp$Feature %<>% gsub("^ASA_[a-zA-Z0-9:punct:]+","ASA",.)
imp$Feature %<>% gsub("^weekday_[a-zA-Z0-9:punct:]+","weekday",.)
imp$Feature %<>% gsub("^BMI[a-zA-Z0-9:punct:_]+","BMI",.)

imp$Feature[imp$Feature %in% Procedure] <- 'Procedure'
imp$Feature[imp$Feature %in% Surgeon] <- 'Surgeon'
imp$Feature[imp$Feature %in% Diag] <- 'Diag'


XGB.imp <- aggregate(. ~ Feature, imp, sum) %>% data.table::as.data.table()

XGB.imp <- XGB.imp[order(XGB.imp$Gain,decreasing = T),]


var=XGB.imp$Feature
imp=XGB.imp$Gain

mplot_importance <- function(var, imp, colours = NA, limit = 23, model_name = NA, subtitle = NA,
                             save = FALSE, file_name = "viz_importance.png", subdir = NA) {
  require(ggplot2)
  require(gridExtra)
  options(warn=-1)
  
  if (length(var) != length(imp)) {
    message("The variables and importance values vectors should be the same length.")
    stop(message(paste("Currently, there are",length(var),"variables and",length(imp),"importance values!")))
  }
  if (is.na(colours)) {
    colours <- c(rep("Categorical",3),"Numerical","Categorical","Categorical","Numerical",
                 "Numerical","Categorical","Numerical","Numerical","Categorical","Numerical","Numerical","Categorical",
                 "Numerical","Categorical","Categorical","Categorical")
  }
  out <- data.frame(var = var, imp = imp, Type = colours)
  if (length(var) < limit) {
    limit <- length(var)
  }
  
  output <- out[1:limit,]
  
  p <- ggplot(output, 
              aes(x = reorder(var, imp), y = imp * 100, 
                  label = round(100 * imp, 1))) + 
    geom_col(aes(fill = Type), width = 0.2) +
    scale_fill_manual('Data type',values = c("Categorical" = "dodgerblue4", "Numerical" = "gold3"))+
    geom_point(aes(colour = Type), size = 8) + 
    scale_color_manual('Data type',values = c("Categorical" = "dodgerblue4", "Numerical" = "gold3"))+
    coord_flip() + xlab('') + theme_minimal() +
    ylab('Importance') + 
    geom_text(hjust = 0.5, size = 3, inherit.aes = TRUE, colour = NA, position=position_dodge(width=5)) +
    labs(title = paste0("Variables Importances in XGB. (", limit, " / ", length(var), " plotted)"))
  
  if (length(unique(output$Type)) == 1) {
    p <- p + geom_col(fill = colours, width = 0.2) +
      geom_point(colour = colours, size = 8) + 
      guides(fill = FALSE, colour = FALSE)  
    # +geom_text(hjust = 0.5, size = 3.5, inherit.aes = TRUE, colour = NA)
  }
  if(!is.na(model_name)) {
    p <- p + labs(caption = model_name)
  }
  if(!is.na(subtitle)) {
    p <- p + labs(subtitle = subtitle)
  }  
  if(save == TRUE) {
    if (!is.na(subdir)) {
      dir.create(file.path(getwd(), subdir))
      file_name <- paste(subdir, file_name, sep="/")
    }
    p <- p + ggsave(file_name, width=6, height=5)
  }
  
  return(p)
  
}

mplot_importance(var=XGB.imp$Feature,imp=XGB.imp$Gain, limit = 23)

##################################################################################
################################ BA plots ########################################
##################################################################################

library(BlandAltmanLeh)
library(ggExtra)
library(ggpubr) 

Pred_data <- read.csv('D:/AI Scheduling/AI-scheduling-for-Surgerry/Data/Pred_data.csv',header = T)

Test <- read.csv('D:/AI Scheduling/AI-scheduling-for-Surgerry/Test.csv',header = T)
Outlier_data <- Test
Outlier_data$Index <- 1:dim(Test)[1]
Outlier_data$Diff <- Pred_data$in_mins - Pred_data$XGB_pred
Outlier_data <- Outlier_data[abs(Outlier_data$Diff) >= 120,]
#write.csv(Outlier_data,'D:/AI Scheduling/AI-scheduling-for-Surgerry/Outlier_data.csv',row.names = FALSE)
#View(Outlier_data)

Y_test <- Pred_data$in_mins
test.pred <- Pred_data$DOC_pred
Delta <- sapply(test.pred,function(x) min(max(c(0.15*x, 15)),60))

DOC_dist_test <-
  ifelse(Y_test > test.pred + Delta,"overage",
         ifelse(Y_test < test.pred - Delta,"underage","within"))
table(DOC_dist_test)/sum(table(DOC_dist_test))

Block <- ifelse((Y_test+test.pred)/2 < 60,'<60',
                ifelse((Y_test+test.pred)/2 < 120,'<120',
                       ifelse((Y_test+test.pred)/2 < 240,'<240', '>=240')))

table(Block,DOC_dist_test)/rowSums(table(Block,DOC_dist_test))


DOC_BAplt <- 
  ggMarginal(
    ggplot(Pred_data, aes(x =  (in_mins+DOC_pred)/2, y = (in_mins-DOC_pred) )) +
      geom_point(aes(color = DOC_dist_test),size=1) +
      scale_color_manual(values=c("azure4", "azure3", "cadetblue4"))+
      geom_abline(slope=0,size = 1, colour = 'gray21',linetype="dashed") +
      geom_vline(xintercept = 0 ,size = 0.6, colour = 'gray30',linetype="dashed") +
      geom_vline(xintercept = 60 ,size = 0.6, colour = 'gray30',linetype="dashed") +
      geom_vline(xintercept = 120 ,size = 0.6, colour = 'gray30',linetype="dashed") +
      geom_vline(xintercept = 240 ,size = 0.6, colour = 'gray30',linetype="dashed") +
      labs(y = "Actual-predicted", x = "") +
      theme_bw() + 
      xlim(0, 600) + ylim(-350, 500) +
      theme(legend.position = "none") 
    , type = "densigram", size=2.5,col = "gray16", fill = "cadetblue3")


sum( (Pred_data$in_mins-Pred_data$DOC_pred)/Pred_data$DOC_pred > 5 )
sum( (Pred_data$in_mins-Pred_data$DOC_pred)/Pred_data$DOC_pred < -1 )

DOC_percent <-
  ggplot(Pred_data, aes(x =  (in_mins+DOC_pred)/2, y = (in_mins-DOC_pred)/DOC_pred )) +
  geom_point(aes(color = DOC_dist_test),size=1) +
  scale_color_manual(values=c("azure4", "azure3", "cadetblue4"))+
  geom_abline(slope=0,size = 1, colour = 'gray21',linetype="dashed") +
  geom_vline(xintercept = 0 ,size = 0.6, colour = 'gray30',linetype="dashed") +
  geom_vline(xintercept = 60 ,size = 0.6, colour = 'gray30',linetype="dashed") +
  geom_vline(xintercept = 120 ,size = 0.6, colour = 'gray30',linetype="dashed") +
  geom_vline(xintercept = 240 ,size = 0.6, colour = 'gray30',linetype="dashed") +
  annotate("text", x=175, y=5, label= "4 outliers above the upper bound",size=5)+
  labs(x = "Actual+Predicted/2") +
  labs(y = "(Actual-Predicted)/Predicted") +
  xlim(0, 600) + ylim(-1, 5) +
  theme_bw() + 
  theme(legend.position = "none")  


Y_test <- Pred_data$in_mins
test.pred <- Pred_data$PCD_pred
Delta <- sapply(test.pred,function(x) min(max(c(0.15*x, 15)),60))

PCD_dist_test <-
  ifelse(Y_test > test.pred + Delta,"overage",
         ifelse(Y_test < test.pred - Delta,"underage","within"))
table(PCD_dist_test)/sum(table(PCD_dist_test))

Block <- ifelse((Y_test+test.pred)/2 < 60,'<60',
                ifelse((Y_test+test.pred)/2 < 120,'<120',
                       ifelse((Y_test+test.pred)/2 < 240,'<240', '>=240')))

table(Block,PCD_dist_test)/rowSums(table(Block,PCD_dist_test))


PCD_BAplt <- 
  ggMarginal(
    ggplot(Pred_data, aes(x =  (in_mins+PCD_pred)/2, y = (in_mins-PCD_pred) )) +
      geom_point(aes(color = PCD_dist_test),size=1) +
      scale_color_manual(values=c("azure4", "azure3", "cadetblue4"))+
      geom_abline(slope=0,size = 1, colour = 'gray21',linetype="dashed") +
      geom_vline(xintercept = 0 ,size = 0.6, colour = 'gray30',linetype="dashed") +
      geom_vline(xintercept = 60 ,size = 0.6, colour = 'gray30',linetype="dashed") +
      geom_vline(xintercept = 120 ,size = 0.6, colour = 'gray30',linetype="dashed") +
      geom_vline(xintercept = 240 ,size = 0.6, colour = 'gray30',linetype="dashed") +
      labs(y = "Actual-predicted",x = "") +
      theme_bw() + 
      xlim(0, 600) + ylim(-350, 500) +
      theme(legend.position = "none") 
    , type = "densigram", size=2.5,col = "gray16", fill = "cadetblue3")


sum((Pred_data$in_mins-Pred_data$PCD_pred)/Pred_data$PCD_pred > 5 )
sum((Pred_data$in_mins-Pred_data$PCD_pred)/Pred_data$PCD_pred < -1 )

PCD_percent <-
  ggplot(Pred_data, aes(x =  (in_mins+PCD_pred)/2, y = (in_mins-PCD_pred)/PCD_pred )) +
  geom_point(aes(color = PCD_dist_test),size=1) +
  scale_color_manual(values=c("azure4", "azure3", "cadetblue4"))+
  geom_abline(slope=0,size = 1, colour = 'gray21',linetype="dashed") +
  geom_vline(xintercept = 0 ,size = 0.6, colour = 'gray30',linetype="dashed") +
  geom_vline(xintercept = 60 ,size = 0.6, colour = 'gray30',linetype="dashed") +
  geom_vline(xintercept = 120 ,size = 0.6, colour = 'gray30',linetype="dashed") +
  geom_vline(xintercept = 240 ,size = 0.6, colour = 'gray30',linetype="dashed") +
  annotate("text", x=175, y=5, label= "6 outliers above the upper bound",size=5)+
  annotate("text", x=175, y=-1, label= "2 outliers below the lower bound",size=5)+
  labs(x = "Actual+Predicted/2") +
  labs(y = "(Actual-Predicted)/Predicted") +
  xlim(0, 600) + ylim(-1, 5) +
  theme_bw() + 
  theme(legend.position = "none") 


Y_test <- Pred_data$in_mins
test.pred <- Pred_data$XGB_pred
Delta <- sapply(test.pred,function(x) min(max(c(0.15*x, 15)),60))

XGB_dist_test <-
  ifelse(Y_test > test.pred + Delta,"overage",
         ifelse(Y_test < test.pred - Delta,"underage","within"))
table(XGB_dist_test)/sum(table(XGB_dist_test))

Block <- ifelse((Y_test+test.pred)/2 < 60,'<60',
                ifelse((Y_test+test.pred)/2 < 120,'<120',
                       ifelse((Y_test+test.pred)/2 < 240,'<240', '>=240')))

table(Block,XGB_dist_test)/rowSums(table(Block,XGB_dist_test))

table(Test$TeamSize,XGB_dist_test)/rowSums(table(Test$TeamSize,XGB_dist_test))
table(Test$Nproced,XGB_dist_test)/rowSums(table(Test$Nproced,XGB_dist_test))
table(Test$TeamSize,XGB_dist_test)
table(Test$Nproced,XGB_dist_test)


XGB_BAplt <- 
  ggMarginal(
    ggplot(Pred_data, aes(x = (in_mins+XGB_pred)/2, y = (in_mins-XGB_pred) )) +
      geom_point(aes(color = XGB_dist_test), size=1) +
      scale_color_manual(values=c("azure4", "azure3", "cadetblue4"))+
      geom_abline(slope=0,size = 1, colour = 'gray21',linetype="dashed") +
      geom_vline(xintercept = 0 ,size = 0.6, colour = 'gray30',linetype="dashed") +
      geom_vline(xintercept = 60 ,size = 0.6, colour = 'gray30',linetype="dashed") +
      geom_vline(xintercept = 120 ,size = 0.6, colour = 'gray30',linetype="dashed") +
      geom_vline(xintercept = 240 ,size = 0.6, colour = 'gray30',linetype="dashed") +
      labs(y = "Actual-predicted",x = "") +
      theme_bw() + 
      #xlim(0, 600) + ylim(-350, 500) +
      theme(legend.position = "none") 
    , type = "densigram", size=2.5,col = "gray16", fill = "cadetblue3")


dim(Pred_data)
sum((Pred_data$in_mins-Pred_data$XGB_pred)/Pred_data$XGB_pred > 5 )
sum((Pred_data$in_mins-Pred_data$XGB_pred)/Pred_data$XGB_pred < -1 )

XGB_percent <-
  ggplot(Pred_data, aes(x = (in_mins+XGB_pred)/2, y = (in_mins-XGB_pred)/XGB_pred  )) +
  geom_point(aes(color = XGB_dist_test),size=1) +
  scale_color_manual(values=c("azure4", "azure3", "cadetblue4"))+
  geom_abline(slope=0,size = 1, colour = 'gray21',linetype="dashed") +
  geom_vline(xintercept = 0 ,size = 0.6, colour = 'gray30',linetype="dashed") +
  geom_vline(xintercept = 60 ,size = 0.6, colour = 'gray30',linetype="dashed") +
  geom_vline(xintercept = 120 ,size = 0.6, colour = 'gray30',linetype="dashed") +
  geom_vline(xintercept = 240 ,size = 0.6, colour = 'gray30',linetype="dashed") +
  labs(x = "Actual+Predicted/2") +
  labs(y = "(Actual-Predicted)/Predicted") +
  xlim(0, 600) + ylim(-1, 5) +
  theme_bw() + 
  theme(legend.position = "none") 



##################################################################################################
##################################### Polar Plots ################################################
##################################################################################################

Test <- read.csv('D:/AI Scheduling/AI-scheduling-for-Surgerry/Data/Test.csv',header=T);dim(Test)
Pred.XGB <- read.csv('D:/AI Scheduling/AI-scheduling-for-Surgerry/Data/Pred_data.csv',header = T)
Pred.XGB <- cbind(Pred.XGB,Test$DivNo)
colnames(Pred.XGB) <- c("in_mins","DOC_pred","PCD_pred","XGB_pred","DivNo")

MEAN <- aggregate(Pred.XGB$in_mins, list(Test$DivNo), mean)$x

Delta <- sapply(Pred.XGB$in_mins,function(x) min(max(c(0.15*x, 15)),60))

Pred.XGB$pred.y <-     
  ifelse(Pred.XGB$in_mins > Pred.XGB$XGB_pred + Delta,"overage",
         ifelse(Pred.XGB$in_mins < Pred.XGB$XGB_pred - Delta,"underage","within"))

table(Pred.XGB$pred.y)/sum(table(Pred.XGB$pred.y))

Pred.XGB2 <- data.frame(aggregate(Pred.XGB$pred.y, by=list(Category=Test$DivNo), FUN=table))
correct <- Pred.XGB2$x[,3]
overage <- Pred.XGB2$x[,1]
underage <- Pred.XGB2$x[,2]

incorrect <- rowSums(Pred.XGB2$x[,c(1,2)])
data <- data.frame(Pred.XGB2$Category,correct,-incorrect,MEAN,-overage,-underage)
colnames(data) <- c('Category','correct', 'incorrect','MEAN','overage','underage')
data$Category <- c('BM','TA','GH','GS','OR','UR','NE', 'CS','TS',
                   'BMS','PS','OG','ENT','PD','OM','DE','OP','PR','CAS','BS','AN','Others')

data <- data %>% mutate(TreeRank = rank(-correct), PopRank = rank(-incorrect)) %>%
  mutate(SqRank = (TreeRank^2)+(PopRank^2)/2) %>% mutate(RankOrder = rank(SqRank))


ggplot(data, aes(x = reorder(Category, TreeRank))) +
  theme(text = element_text(size=16))+
  geom_col(aes(y = correct), fill = "cadetblue3") +
  geom_text(aes(y = 40, label = round(correct, 2)), size = 3)+
  
  geom_col(aes(y = overage+underage), fill = "azure2") +
  geom_text(aes(y = ifelse(data$overage < -40, data$overage - 40, -100) , label = -round(underage, 1)), size = 3)+
  
  geom_col(aes(y = overage), fill = "azure3") +
  geom_text(aes(y = -40, label = -round(overage, 1)), size = 3)+
  geom_point(data = data, aes(y = round(MEAN)), stat='identity', colour = 'black', size = 1.5) +
  coord_polar() +
  theme_minimal()



library(ggplot2)
library(ggthemes)
library(extrafont)
library(plyr)
library(scales)

##################################################################################################
##################################### Bar Plots ################################################
##################################################################################################


charts.data <-  read.csv("D:/AI Scheduling/AI-scheduling-for-Surgerry/Data/Percentage.csv",header=T, encoding = 'UTF-8') 
colnames(charts.data) <- c('Method', 'type', 'Percentage')

charts.data$pos <- c(85,87,88,87,87,87,43,55,59,62,64,63,10,17,20,23,25,24)
charts.data <- charts.data[order(charts.data$type),]


fill <- c("#5F9EA0", "#E1B378","#b2d183")

charts.data$Method <- factor(charts.data$Method, levels = c("Surgeon-based", "Procedure-based", "Reg",
                                                            "LogReg","RF","XGB"))

p4 <-  
  ggplot( aes(x = Method, y = Percentage, fill = type, order=Method), data=charts.data) +
  geom_col(position = position_stack(reverse = TRUE))+
  geom_bar(stat="identity") +
  geom_text(data=charts.data, aes(x = Method, y = pos, label = paste0(round(Percentage),"%")),
            colour="black", size=4) +
  theme(legend.position="bottom", legend.direction="horizontal",
        legend.title = element_blank()) +
  scale_y_continuous(labels = dollar_format(suffix = "%", prefix = "")) +
  labs(x="Model", y="Percentage (%)")+
  scale_fill_manual("legend", values = c("Overage" = "azure3", "Underage" = "azure2", "Within" = "cadetblue3"))+
  theme(axis.line = element_line(size=1, colour = "black"),
        panel.grid.major = element_line(colour = "#d3d3d3"), panel.grid.minor = element_blank(),
        panel.border = element_blank(), panel.background = element_blank()) +
  theme(plot.title = element_text(size = 14, family = "Tahoma", face = "bold"),
        text = element_text(family="Tahoma",size = 16),
        axis.text.x=element_text(colour="black", size = 10),
        axis.text.y=element_text(colour="black", size = 10))
p4


par(mar = c(5,5,1,1))
Rsq <- c(26, 74, 78, 82, 79, 84)
colors <- c("royalblue4", "royalblue4", "cornflowerblue", "cornflowerblue", "navyblue", "navyblue")
names <- c("Surgeon-\nbased","Procedure-\nbased","Reg","LogReg","RF","XGB")

bp <- barplot(Rsq, ylim=c(0,100), las=1, col=colors, names=names ,ylab=as.expression(bquote(R^2 ~ " (%)"))
              ,space=0.5,xlab = 'Model',cex=1, cex.lab = 1, cex.axis = 1,font=3, lwd = 2,cex.lab=1.5)
labels <-c("26%\n(MAE = 67.4)", "74%\n(MAE = 36.8)", "78%\n(MAE = 33.6)", "82%\n(MAE = 33.3)", "79%\n(MAE = 31)", "84%\n(MAE = 31.1)")
text(bp, Rsq+2, labels, cex=1, pos=3,font=3)


