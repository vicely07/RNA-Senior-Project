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
saveRDS(dt.fit, "DT4.rds")
Pred <- predict(dt.fit,testing)

cm<- confusionMatrix(Pred,testing$class)
print("CM for DT:")
print(cm)

print("done saving rds decision tree")

######################################################################################################
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
dt.fit <- do.DT(training)
saveRDS(dt.fit, "DT6.rds")
Pred <- predict(dt.fit,testing)

cm<- confusionMatrix(Pred,testing$class)
print("CM for DT:")
print(cm)

print("done saving rds decision tree")
#############################################################################################################
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
dt.fit <- do.DT(training)
saveRDS(dt.fit, "DT8.rds")
Pred <- predict(dt.fit,testing)

cm<- confusionMatrix(Pred,testing$class)
print("CM for DT:")
print(cm)

print("done saving rds decision tree")