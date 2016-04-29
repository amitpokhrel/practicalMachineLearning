# Practical Machine Learning Course Project
Amit Pokhrel  
April 29, 2016  
Executive Summary
===
Data collection about personal activity has been more easy with the availability of devices such as Jawbone Up, Nike FuelBand, and Fitbit. These sort of devices are used to record self movement. In this project , I will try and analyze dataset from the measurements of activities by group of enthusiasts. I will use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. 

The goal of this project is to predict the manner in which they did the exercise. This is the "classe" variable in the training set. You may use any of the other variables to predict with. You should create a report describing how you built your model, how you used cross validation, what you think the expected out of sample error is, and why you made the choices you did. You will also use your prediction model to predict 20 different test cases.

Loading libraries
===

```r
library(caret)
```

```
## Warning: package 'caret' was built under R version 3.2.5
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```
## Warning: package 'ggplot2' was built under R version 3.2.5
```

```r
library(gbm)
```

```
## Warning: package 'gbm' was built under R version 3.2.5
```

```
## Loading required package: survival
```

```
## 
## Attaching package: 'survival'
```

```
## The following object is masked from 'package:caret':
## 
##     cluster
```

```
## Loading required package: splines
```

```
## Loading required package: parallel
```

```
## Loaded gbm 2.1.1
```

Loading Dataset
===
Dataset to develop model and validate model is downloaded from provided link.

The training data for this project are available here:
[training dataset](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv)

The test data are available here:
[testing dataset](https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv)

To load data to R, you can download it manually or by using following commands:

```r
train_file <- "pml-training.csv"
test_file <- "pml-testing.csv" 
train_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
if (!file.exists(train_file)){
    download.file(train_url)    
}

test_url <- "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
if (!file.exists(test_file)){
    download.file(test_url)  
}
```

After downloading the dataset, load the dataset into R

```r
train_data <- read.csv(train_file, na.strings = c("#DIV/0!","NA"))
final_test_data <- read.csv(test_file, na.strings = c("#DIV/0!","NA"))
```

# Cleaning Data
First five columns(X,user_name,raw_timestamp_part_1, raw_timestamp_part_2, cvtd_timestamp) has no significance in building a prediction model. So, remove first five columns

```r
train_data <- subset(train_data, select = -(1:5))

# remove variables with nearly zero variance
zerovarIndex <- nearZeroVar(train_data)
train_data <- train_data[, -zerovarIndex]

# remove variables that are almost always NA
mostlyNA <- sapply(train_data, function(x) mean(is.na(x))) > 0.9
train_data <- train_data[, mostlyNA == F]
```

Model Building
===
I decided to use RandomForest model to see if it returns acceptable performance. I will be using `train` function in `caret` package to train the model and use 10-fold cross validation.

```r
#partition the dataset into train and test set
dataIndex <- createDataPartition(train_data$classe, p = 0.7, list = FALSE)
training_set <- train_data[dataIndex,]
testing_set <- train_data[-dataIndex,]

modelcontrol <- trainControl(method = "cv", number = 10, verboseIter = FALSE)
rfFit <- train(classe ~ ., method = "rf", data = training_set, trControl = modelcontrol)
```

```
## Loading required package: randomForest
```

```
## Warning: package 'randomForest' was built under R version 3.2.5
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

Lets use boosting algorithm with 10-fold cross validation to predict classe.

```r
boostFit <- train(classe ~ ., method = "gbm", data = training_set, verbose = FALSE, trControl = modelcontrol)
```

```
## Loading required package: plyr
```

```
## Warning: package 'plyr' was built under R version 3.2.4
```

Random Forest vs Boosting Model Evaluation
===
Use the fitted model to predict the classe in testing dataset. Confusion matrix will compare predicted vs actual values.

```r
plot(rfFit, ylim = c(0.9, 1), main = "Random Forest model")
```

![](Machine_learning_assignment_files/figure-html/Fitevaluation-1.png)

```r
plot(boostFit, ylim = c(0.9, 1), main = "Boosting model")
```

![](Machine_learning_assignment_files/figure-html/Fitevaluation-2.png)

```r
# use the random forest model fitted to predict classe in testing set
rfFit_predicted <- predict(rfFit, newdata = testing_set)

# show confusion matrix to get estimate of out-of-sample error from prediction
confusionMatrix(testing_set$classe, rfFit_predicted)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1674    0    0    0    0
##          B    4 1131    3    1    0
##          C    0    3 1020    3    0
##          D    0    0    4  960    0
##          E    0    0    0    3 1079
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9964          
##                  95% CI : (0.9946, 0.9978)
##     No Information Rate : 0.2851          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9955          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9976   0.9974   0.9932   0.9928   1.0000
## Specificity            1.0000   0.9983   0.9988   0.9992   0.9994
## Pos Pred Value         1.0000   0.9930   0.9942   0.9959   0.9972
## Neg Pred Value         0.9991   0.9994   0.9986   0.9986   1.0000
## Prevalence             0.2851   0.1927   0.1745   0.1643   0.1833
## Detection Rate         0.2845   0.1922   0.1733   0.1631   0.1833
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9988   0.9978   0.9960   0.9960   0.9997
```

```r
# use the boosting model fitted to predict classe in testing set
boostFit_predicted <- predict(boostFit, newdata = testing_set)

# show confusion matrix to get estimate of out-of-sample error from prediction
confusionMatrix(testing_set$classe, boostFit_predicted)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1672    2    0    0    0
##          B   14 1104   18    3    0
##          C    0    9 1015    2    0
##          D    0    3    8  953    0
##          E    0    4    0    7 1071
## 
## Overall Statistics
##                                          
##                Accuracy : 0.9881         
##                  95% CI : (0.985, 0.9907)
##     No Information Rate : 0.2865         
##     P-Value [Acc > NIR] : < 2.2e-16      
##                                          
##                   Kappa : 0.985          
##  Mcnemar's Test P-Value : NA             
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9917   0.9840   0.9750   0.9876   1.0000
## Specificity            0.9995   0.9927   0.9977   0.9978   0.9977
## Pos Pred Value         0.9988   0.9693   0.9893   0.9886   0.9898
## Neg Pred Value         0.9967   0.9962   0.9946   0.9976   1.0000
## Prevalence             0.2865   0.1907   0.1769   0.1640   0.1820
## Detection Rate         0.2841   0.1876   0.1725   0.1619   0.1820
## Detection Prevalence   0.2845   0.1935   0.1743   0.1638   0.1839
## Balanced Accuracy      0.9956   0.9883   0.9864   0.9927   0.9989
```

From above comparison, random forest is the best model that can be used to fit the dataset. 
Out of Sample error
===

```r
## Calculate OOS Error
missClass = function(values, predicted) {
        sum(predicted != values) / length(values)
}
OOS_errRateRF = missClass(testing_set$classe, rfFit_predicted)
OOS_errRateRF
```

```
## [1] 0.003568394
```

Estimated out of sample error rate for the random forests model is 0.0035684 as reported by the final model.

Final Prediction
===
Finally, predicting the classe of testing dataset provided using the model selected and writing the result to files.

```r
# predict on test set
preds <- predict(rfFit, newdata = final_test_data)

# convert predictions to character vector
preds <- as.character(preds)

# create function to write predictions to files
pml_write_files <- function(x) {
    n <- length(x)
    for (i in 1:n) {
        filename <- paste0("predicted_output/problem_id_", i, ".txt")
        write.table(x[i], file = filename, quote = FALSE, row.names = FALSE, col.names = FALSE)
    }
}

# create prediction files to submit
pml_write_files(preds)
```
