# training = read.csv("Kmer_Train/kmer_6_train.csv", header = TRUE)
# training <- training[,names(training) != "ID"]
# head(training)
# training <- training[sample(nrow(training), nrow(training)), ] #randomizes the rows
# training$class[training$class == "1"] <- "positive"
# training$class[training$class == "0"] <- "negative"
# training$class <- factor(training$class)


# #Preparing testing data

# testing = read.csv("C:/Users/Mango/Documents/KmerResearch/Kmer_Test/kmer_8_test.csv", header = TRUE)
# testing <- read.table(file,sep=",",header=TRUE)
# testing <- testing[,names(testing) != "ID"]
# testing <- testing[sample(nrow(testing), nrow(testing)), ] #randomizes the rows
# testing$class[testing$class == "1"] <- "positive"
# testing$class[testing$class == "0"] <- "negative"
# testing$class <- factor(testing$class)
# print("#########test and train loaded#############")
# install.packages("caret")
# install.packages("e1071")
# #install.packages("glmnet")
# #install.packages("C50")
# #suppressMessages(library(C50))
suppressMessages(library(caret))
suppressMessages(library(e1071))
# print("random forest training")
# # CARET Random Forest

do.RF <- function(training)
{  
 set.seed(313)
 n <- dim(training)[2]
 gridRF <- expand.grid(mtry = seq(from=0,by=as.integer(n/10),to=n)[-1]) #may need to change this depend on your data size
 ctrl.crossRF <- trainControl(method = "cv",number = 10,classProbs = TRUE,savePredictions = TRUE,allowParallel=TRUE)
 rf.Fit <- train(class ~ .,data = training,method = "rf",metric = "Accuracy",preProc = c("center", "scale"),
                 ntree = 200, tuneGrid = gridRF,trControl = ctrl.crossRF)
 rf.Fit
}


# predict using tuned random forest
# Pred <-  predict(rf.Fit,testing)
# print("confusion matrix")
# cm <- confusionMatrix(Pred,testing$class)
# print("CM for RF:") 
# print(cm)

# print("done saving rds randomforest")

#Regularization elastic-net logistic regression:
#install.packages("glmnet", repos = "http://cran.us.r-project.org")

training = read.csv("Kmer_Train/kmer_8_train.csv", header = TRUE)
training <- training[,names(training) != "ID"]
training <- training[sample(nrow(training), nrow(training)), ] #randomizes the rows
training$class[training$class == "1"] <- "positive"
training$class[training$class == "0"] <- "negative"
training$class <- factor(training$class)

#CARET Random forest
print("random forest training")
rf.Fit <- do.RF(training)
print("done training")
saveRDS(rf.Fit, "RF8.Rds")
print("saveRDS")


# testing = read.csv("Kmer_Test/kmer_8_test.csv", header = TRUE)
# testing <- read.table(file,sep=",",header=TRUE)
# testing <- testing[,names(testing) != "ID"]
# testing <- testing[sample(nrow(testing), nrow(testing)), ] #randomizes the rows
# testing$class[testing$class == "1"] <- "positive"
# testing$class[testing$class == "0"] <- "negative"
# testing$class <- factor(testing$class)
# print("#########test and train loaded#############")

###################MCC###################
mcc <- function (conf_matrix)
{
        TP <- conf_matrix$table[1,1]
        TN <- conf_matrix$table[2,2]
        FP <- conf_matrix$table[1,2]
        FN <- conf_matrix$table[2,1]
        
        mcc_num <- (TP*TN - FP*FN)
        mcc_den <- 
                as.double((TP+FP))*as.double((TP+FN))*as.double((TN+FP))*as.double((TN+FN))
        
        mcc_final <- mcc_num/sqrt(mcc_den)
        return(mcc_final)
}

# CARET Random Forest

rf.Fit <- do.RF(training)
saveRDS(rf.Fit, "RF.Rds")
print(rf.Fit)

rf.Fit <- readRDS("C:/Users/Mango/Documents/KmerResearch/KNN8.rds")
#predict using tuned random forest
Pred <-  predict(rf.Fit,testing)
print("confusion matrix")
cm <- confusionMatrix(factor(Pred),testing$class)
print("CM for RF:") 
print(cm)
mcc = mcc(cm)
print(mcc)
y <- factor(testing$class)
precision <- posPredValue(predictions, y, positive="positive")
recall <- sensitivity(predictions, y, positive="positive")
F1 <- (2 * precision * recall) / (precision + recall)


library(pROC)
Pred <-  predict(rf.Fit, testing, type = "prob")
result.roc <- plot(roc(testing$class, Pred$positive))
#plot(result.roc, print.thres="best", print.thres.best.method="closest.topleft")
auc <- auc(result.roc)
print(cm$overall)
print(auc)
print(mcc)
print(precision)
print(recall)
print(F1)
