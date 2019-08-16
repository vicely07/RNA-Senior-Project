training = read.csv("Kmer_Train/kmer_4_train.csv", header = TRUE)
training <- training[,names(training) != "ID"]
head(training)
training <- training[sample(nrow(training), nrow(training)), ] #randomizes the rows
training$class[training$class == "1"] <- "positive"
training$class[training$class == "0"] <- "negative"
training$class <- factor(training$class)


#Preparing testing data

testing = read.csv("Kmer_Test/kmer_4_test.csv", header = TRUE)
#testing <- read.table(file,sep=",",header=TRUE)
testing <- testing[,names(testing) != "ID"]
testing <- testing[sample(nrow(testing), nrow(testing)), ] #randomizes the rows
testing$class[testing$class == "1"] <- "positive"
testing$class[testing$class == "0"] <- "negative"
testing$class <- factor(testing$class)
print("#########test and train loaded#############")
#install.packages("caret")
#install.packages("e1071")
#install.packages("glmnet")
#install.packages("C50")
#suppressMessages(library(C50))
suppressMessages(library(caret))
suppressMessages(library(e1071))
print("random forest training")

#Load R libraries for model generation
suppressMessages(library(kknn))
#CARET KNN:
grid = expand.grid(kmax=c(1:20),distance=2,kernel="optimal")
ctrl.cross <- trainControl(method="cv",number=10, classProbs=TRUE,savePredictions=TRUE)
print("KNN training")
#Requires package 'kknn' to run
knnFit.cross <- train(class ~ .,
data = training, # training data
method ="kknn",  # model  
metric="Accuracy", #evaluation metric
preProc=c("center","scale"), # data to be scaled
tuneGrid = grid, # range of parameters to be tuned
trControl=ctrl.cross) # training controls
#print(knnFit.cross)
#plot(knnFit.cross)
#Fifth, Perform predictions on the testing set, and confusion matrix. Accuracies on testing and training should be similar.

Pred <- predict(knnFit.cross,testing)
cm<- confusionMatrix(Pred,testing$class)
print("CM for KNN:")
print(cm)
saveRDS(knnFit.cross, "KNN4.rds")
print("done saving rds KNN")
#######################################################################################################
training = read.csv("Kmer_Train/kmer_6_train.csv", header = TRUE)
training <- training[,names(training) != "ID"]
head(training)
training <- training[sample(nrow(training), nrow(training)), ] #randomizes the rows
training$class[training$class == "1"] <- "positive"
training$class[training$class == "0"] <- "negative"
training$class <- factor(training$class)


#Preparing testing data

testing = read.csv("Kmer_Test/kmer_6_test.csv", header = TRUE)
#testing <- read.table(file,sep=",",header=TRUE)
testing <- testing[,names(testing) != "ID"]
testing <- testing[sample(nrow(testing), nrow(testing)), ] #randomizes the rows
testing$class[testing$class == "1"] <- "positive"
testing$class[testing$class == "0"] <- "negative"
testing$class <- factor(testing$class)

Pred <- predict(knnFit.cross,testing)
cm<- confusionMatrix(Pred,testing$class)
print("CM for KNN:")
print(cm)
saveRDS(knnFit.cross, "KNN6.rds")
print("done saving rds KNN")
##############################################################################################################
training = read.csv("Kmer_Train/kmer_8_train.csv", header = TRUE)
training <- training[,names(training) != "ID"]
head(training)
training <- training[sample(nrow(training), nrow(training)), ] #randomizes the rows
training$class[training$class == "1"] <- "positive"
training$class[training$class == "0"] <- "negative"
training$class <- factor(training$class)


#Preparing testing data

testing = read.csv("Kmer_Test/kmer_8_test.csv", header = TRUE)
#testing <- read.table(file,sep=",",header=TRUE)
testing <- testing[,names(testing) != "ID"]
testing <- testing[sample(nrow(testing), nrow(testing)), ] #randomizes the rows
testing$class[testing$class == "1"] <- "positive"
testing$class[testing$class == "0"] <- "negative"
testing$class <- factor(testing$class)

Pred <- predict(knnFit.cross,testing)
cm<- confusionMatrix(Pred,testing$class)
print("CM for KNN:")
print(cm)
saveRDS(knnFit.cross, "KNN8.rds")
print("done saving rds KNN")