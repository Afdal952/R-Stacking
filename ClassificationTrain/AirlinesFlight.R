#Perform Some Classification Algorithms on AirlinesFlight Data Set
#1. LOAD THE LIBRARIES#
library(caret)
library(dplyr)
library(ggplot2)
library(e1071)
library(randomForest)
library(nnet)
library(xgboost)
library(rpart)
library(class)
library(naivebayes)
library(caretEnsemble)

#2. LOAD THE DATA#
#Open the working directory on D:/Projects-R
setwd("D:/Projects-R")
airlines <- read.csv("Data/AirlinesFlight/airlines_flights_data.csv")
View(airlines)

#3. DATA PREPROCESSING#
#Check for missing values
sum(is.na(airlines))

#Check the structure of the data
str(airlines)

#Convert all categorical variables to factors
categorical_vars <- c("airline", "flight", "source_city",
"departure_time", "stops", "arrival_time", "destination_city", "class") 
airlines[categorical_vars] <- lapply(airlines[categorical_vars], as.factor)

str(airlines)

#Split data into train and test#
set.seed(123) # For reproducibility
train_index <- sample(1:nrow(airlines), 0.75 * nrow(airlines))
train_data <- airlines[train_index, ]
test_data <- airlines[-train_index, ]

#4. APPLY CLASSIFICATION ALGORITHMS#
##4.1. Logistic Regression##
logistic_model <- glm(class ~ ., data = train_data, family = binomial)
summary(logistic_model)

#Predict on test data
logistic_pred <- predict(logistic_model, newdata = test_data, type = "response")
logistic_pred_class <- ifelse(logistic_pred > 0.5, "Business", "Economy")

#Confusion Matrix
confmat = table(test_data$class, logistic_pred_class)
confmat$Accuracy = sum(diag(confmat)) / sum(confmat)
print(confmat)

##4.2. Decision Tree##
##4.3. Random Forest##
##4.4. Support Vector Machine##
##4.5. XGBoost##
##4.6. Neural Networks##
##4.7. K-Nearest Neighbors##
##4.8. Naive Bayes##
##4.9. Stacking Ensemble Method##
