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
install.packages("C50")
suppressMessages(library(C50))
suppressMessages(library(caret))
suppressMessages(library(e1071))
print("random forest training")

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
saveRDS(boost.Fit, "Boost4.rds")
print("done saving rds boosted tree")

######################################################################################################################
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
print("boost training")
#CARET boosted trees
boost.Fit <- do.Boost(training)
print(boost.Fit)
Pred <-  predict(boost.Fit,testing)
cm <- confusionMatrix(Pred,testing$class)
print("CM for Boosted:")
print(cm)
saveRDS(boost.Fit, "Boost6.rds")
print("done saving rds boosted tree")

###############################################################################################################################
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
print("boost training")
#CARET boosted trees
boost.Fit <- do.Boost(training)
print(boost.Fit)
Pred <-  predict(boost.Fit,testing)
cm <- confusionMatrix(Pred,testing$class)
print("CM for Boosted:")
print(cm)
saveRDS(boost.Fit, "Boost8.rds")
print("done saving rds boosted tree")