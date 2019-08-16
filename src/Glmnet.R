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
install.packages("glmnet")
library(glmnet)
x <- training
x$class <- NULL
x <- as.matrix(x)
y <- training$class
print("training for logistic regession")
logreg.fit = glmnet(x, y)
Pred <- predict(logreg.fit,testing)
cm<- confusionMatrix(Pred,testing$class)
print("CM for Log Reg:")
print(cm)
saveRDS(do.DT, "LogReg4.rds")
print("done saving rds reg")

####################################################################################
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
x <- training
x$class <- NULL
x <- as.matrix(x)
y <- training$class
print("training for logistic regession")
logreg.fit = glmnet(x, y)
Pred <- predict(logreg.fit,testing)
cm<- confusionMatrix(Pred,testing$class)
print("CM for Log Reg:")
print(cm)
saveRDS(do.DT, "LogReg6.rds")
print("done saving rds reg")
#########################################################################################
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
x <- training
x$class <- NULL
x <- as.matrix(x)
y <- training$class
print("training for logistic regession")
logreg.fit = glmnet(x, y)
Pred <- predict(logreg.fit,testing)
cm<- confusionMatrix(Pred,testing$class)
print("CM for Log Reg:")
print(cm)
saveRDS(do.DT, "LogReg8.rds")
print("done saving rds reg")
