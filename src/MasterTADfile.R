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
#CARET Random Forest

#do.RF <- function(training)
#{  
#  set.seed(313)
#  n <- dim(training)[2]
#  gridRF <- expand.grid(mtry = seq(from=0,by=as.integer(n/10),to=n)[-1]) #may need to change this depend on your data size
#  ctrl.crossRF <- trainControl(method = "cv",number = 10,classProbs = TRUE,savePredictions = TRUE,allowParallel=TRUE)
#  rf.Fit <- train(class ~ .,data = training,method = "rf",metric = "Accuracy",preProc = c("center", "scale"),
#                  ntree = 200, tuneGrid = gridRF,trControl = ctrl.crossRF)
#  rf.Fit
#}

#CARET Random forest
#rf.Fit <- do.RF(training)
#saveRDS(rf.Fit, "RF.Rds")
#print(rf.Fit)
#predict using tuned random forest
#Pred <-  predict(rf.Fit,testing)
#print("confusion matrix")
#cm <- confusionMatrix(Pred,testing$class)
#print("CM for RF:") 
#print(cm)

#print("done saving rds randomforest")

#Regularization elastic-net logistic regression:
#install.packages("glmnet", repos = "http://cran.us.r-project.org")

library(glmnet)
x <- training
x$class <- NULL
x <- as.matrix(x)
x
y <- training$class
y
print("training for logistic regession")
logreg.fit = glmnet(x, y)
Pred <- predict(logreg.fit,testing)
cm<- confusionMatrix(Pred,testing$class)
print("CM for Log Reg:")
print(cm)
saveRDS(do.DT, "LogReg.rds")
print("done saving rds reg")

#CARET Decision Tree:
#this is based on CARET, but sometimes doesn't run well, use the e1071 instead
print("decision tree training")
do.DT <- function(training)
{
  set.seed(1)
  grid <- expand.grid(cp = 2^seq(from = -30 , to= 0, by = 2) )
  ctrl.cross <- trainControl(method = "cv", number = 5,classProbs = TRUE)
  dec_tree <-   train(class ~ ., data= training,perProc = c("center", "scale"),
                      method = 'rpart', #rpart for classif. dec tree
                      metric ='Accuracy',
                      tuneGrid= grid, trControl = ctrl.cross
  )
  dec_tree
}
dt.fit <- do.DT(training)
saveRDS(dt.fit, "DT.rds")
Pred <- predict(dt.fit,testing)

cm<- confusionMatrix(Pred,testing$class)
print("CM for DT:")
print(cm)

print("done saving rds decision tree")

suppressMessages(library(C50))
#This is an example of CARET boosted trees using C50.
do.Boost <- function(training)
{ 
  #trials = number of boosting iterations, or (simply number of trees)
  #winnow = remove unimportant predictors
  gridBoost <- expand.grid(model="tree",trials=seq(from=1,by=2,to=100),winnow=FALSE)
  set.seed(1)
  C5.0.Fit <- train(class ~ .,data = training,method = "C5.0",metric = "Accuracy",preProc = c("center", "scale"),
                    tuneGrid = gridBoost,trControl = ctrl.crossBoost)
  
  C5.0.Fit
}

print("boost training")
#CARET boosted trees
boost.Fit <- do.Boost(training)
print(boost.Fit)
Pred <-  predict(boost.Fit,testing)
cm <- confusionMatrix(Pred,testing$class)
print("CM for Boosted:")
print(cm)
saveRDS(boost.Fit, "Boost.rds")
print("done saving rds boosted tree")

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
saveRDS(knnFit.cross, "KNN.rds")
print("done saving rds KNN")