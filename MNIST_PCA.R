setwd("~/R Code ML CW")

library(class)
library(dplyr)
library(ggfortify)
library(graphics)


#Read mnist 

load_mnist <- function() {
  load_image_file <- function(filename) {
    ret = list()
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    ret$n = readBin(f,'integer',n=1,size=4,endian='big')
    nrow = readBin(f,'integer',n=1,size=4,endian='big')
    ncol = readBin(f,'integer',n=1,size=4,endian='big')
    x = readBin(f,'integer',n=ret$n*nrow*ncol,size=1,signed=F)
    ret$x = matrix(x, ncol=nrow*ncol, byrow=T)
    close(f)
    ret
  }
  load_label_file <- function(filename) {
    f = file(filename,'rb')
    readBin(f,'integer',n=1,size=4,endian='big')
    n = readBin(f,'integer',n=1,size=4,endian='big')
    y = readBin(f,'integer',n=n,size=1,signed=F)
    close(f)
    y
  }
  train <<- load_image_file('train-images-idx3-ubyte')
  test <<- load_image_file('t10k-images-idx3-ubyte')
  
  train$y <<- load_label_file('train-labels-idx1-ubyte')
  test$y <<- load_label_file('t10k-labels-idx1-ubyte')  
}


#Load MNIST DATASET 
load_mnist()

#Checking summary and dimensions 
summary(train$x)
dim(train$x)

summary(test$x)
dim(test$x)


#Check for missing values in training data 
train_rows <- nrow(train$x)
complete_train  <- sum(complete.cases(train$x))
complete_train_rows <- complete_train/train_rows
complete_train_rows

#Check for missing values in testing data 
test_rows <- nrow(test$x)
complete_test <- sum(complete.cases(test$x))
complete_test_rows <- complete_test/test_rows
complete_test_rows


#Check for non-numerical values
non_numerical_train <-str(train$x)
non_numerical_test <-str(test$x)

#Check for NA Values
which(is.na(train$x))
which(is.na(train$y))
which(is.na(test$x))
which(is.na(test$y))

# Normalization for both data sets
Xtrain_data <- train$x / 255
Xtest_data <- test$x /255

#Center data in both data sets
Xtrain_data <- as.data.frame(scale(Xtrain_data, scale = FALSE, center = TRUE))
Xtest_data <- as.data.frame(scale(Xtest_data, scale = FALSE, center = TRUE))

# Perform PCA
Timer  <- proc.time()
pca_train <- prcomp(Xtrain_data)
knn_PCA_Timer <- proc.time() - Timer 
knn_PCA_Timer

#Compute STD_DEV
standard_deviation <- pca_train$sdev

#Compute Variance
pca_variance <- standard_deviation ^2

#Compute variance of first 20 components 
pca_variance[1:20]


#proportion of variance explained 
proportion_variance <- pca_variance/sum(pca_variance)

#Compute first 20 proportions 
proportion_variance[1:20] 

#Plot to track the data with the most variability 
plot(proportion_variance, xlab = "Pcomp", ylab = "Proportion of Variance" , type = "b")

#Plotting a cumulative variance plot to give a picture of components
plot(cumsum(proportion_variance), xlab = "Prcomp" , ylab = "Cumulative Proportion of Variance Exp" , type = "b")

screeplot(pca_train)

autoplot(pca_train, data = Xtrain_data)


most_variance <- as.data.frame(pca_variance/sum(pca_variance))

most_variance <- cbind(c(1:784), cumsum(most_variance))

colnames(most_variance) <- c("Principal Component" , "Cumulative Sum of Variance")

head(most_variance,44)


Xtrain_data <- as.data.frame(pca_train$x)
Ytrain_data <- as.factor(train$y)
Ytest_data <- as.factor(test$y)

#Apply PCA training data transformation on the testing data
Timer  <- proc.time()
Xtest_data <- predict(pca_train, newdata = Xtest_data)
Test_Transform_Timer <- proc.time() - Timer 
Test_Transform_Timer

Xtrain_data <- as.data.frame(Xtrain_data[,1:44])
Xtest_data <- as.data.frame(Xtest_data[,1:44])



#-------------------------------------KNN CLASSIFIER --------------------------------
i=1 
K_loop = 1 
for(i in 1:10){
  Knn_fit_Timer <- proc.time()
  knn_prediction <- knn(train= Xtrain_data, test=Xtest_data, cl = Ytrain_data, k=i)
  proc.time() - Knn_fit_Timer
  Knn_fit_Timer
  knn_conf_matrix <- table("Real Class" = Ytest_data, "Predicted Class" = knn_prediction)
  knn_conf_matrix
  K_loop[i]  <- sum(diag(knn_conf_matrix)) / NROW(Ytest_data)
  k  = i 
  cat(k,"=",K_loop[i],'\n')
  print(knn_conf_matrix)
  
}


plot(K_loop , type = "b" , xlab ="K", ylab= "Acc")
plot(100-K_loop , type = "b" , xlab = "K", ylab= "Error Rate for each K value")



#---------------------------------Creating a Random Forest Model-----------------------------
RF_train <- data.frame(Xtrain_data,label= Ytrain_data)
RF_test <- data.frame(Xtest_data,label = Ytest_data)

library(randomForest)
set.seed(22)
RF_fit_Timer <- proc.time()
RFmodel <- randomForest(RF_train$label~ ., data=RF_train[,-1])
print(RFmodel)
plot(RFmodel)
proc.time() - RF_fit_Timer

rForest_prediction <- predict(RFmodel, newdata = RF_test[,-1])

plot(rForest_prediction)

RF_conf_matrix <- table(RF_test$label,rForest_prediction)
RF_conf_matrix

R_acc <- sum(diag(RF_conf_matrix)) / nrow(RF_test)
R_acc


