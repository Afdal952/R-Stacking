#Perform Some Classification Algorithms on SULSELHealth DataSet
#1. LOAD THE LIBRARIES####
library(caret)
library(readxl)
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
library(gbm)

#2. LOAD THE DATA####
#Open the working directory on D:/Projects-R
setwd("D:/Projects-R")
SULSEL <- read_excel("Data/SULSELL.xlsx")
View(SULSEL)

#3. DATA PREPROCESSING####
#Check for missing values
sum(is.na(SULSEL))

#Check the structure of the data
str(SULSEL)

#Convert all categorical variables to factors
categorical_vars <- c("STRATA", "provkab", "B4K8", "B4K9", "B07A", "G11", "G23A", "G23D", "G23F", "G23G", "G23H", "G23J", "G31K", "G32") 
SULSEL[categorical_vars] <- lapply(SULSEL[categorical_vars], as.factor)

str(SULSEL)

#3. DATA PREPROCESSING####
#Check for missing values
sum(is.na(SULSEL))

#Check the structure of the data
str(SULSEL)

#Convert all categorical variables to factors
categorical_vars <- c("Gender", "Sleep_Quality", "Stress_Level", "Health_Issues", "Occupation") 
SULSEL[categorical_vars] <- lapply(SULSEL[categorical_vars], as.factor)

str(SULSEL)

#Checking the balance of the Categories on G32 Variable
table(SULSEL$G32)

#Split data into train and test#
set.seed(123) # For reproducibility
train_index <- sample(1:nrow(SULSEL), 0.80 * nrow(SULSEL))
train_data <- SULSEL[train_index, ]
test_data <- SULSEL[-train_index, ]

#APPLY ML ALGORITHM
#RANDOMFOREST
set.seed(123)
rf_model <- randomForest(G32 ~ ., data = train_data, importance = TRUE, ntree = 500)
rf_predictions <- predict(rf_model, newdata = test_data)
rf_confusion <- confusionMatrix(rf_predictions, test_data$G32)
print(rf_confusion)
varImpPlot(rf_model)

#XGBOOST
set.seed(123)
train_matrix <- model.matrix(G32 ~ . - 1, data = train_data)
train_label <- as.numeric(train_data$G32) - 1
test_matrix <- model.matrix(G32 ~ . - 1, data = test_data)
test_label <- as.numeric(test_data$G32) - 1

dtrain <- xgb.DMatrix(data = train_matrix, label = train_label)
dtest <- xgb.DMatrix(data = test_matrix, label = test_label)

xgb_params <- list(objective = "multi:softmax", num_class = length(levels(train_data$G32)), eval_metric = "mlogloss")
xgb_model <- xgb.train(params = xgb_params, data = dtrain, nrounds = 100)
xgb_predictions <- predict(xgb_model, newdata = dtest)
# Get the original factor levels
original_levels <- levels(test_data$G32)

# Map the numeric predictions back to the original factor levels
# We add 1 because xgb_predictions are 0-indexed, but R indexing is 1-indexed
predicted_levels <- factor(original_levels[xgb_predictions + 1], levels = original_levels)

# Now create the confusion matrix with matching factor levels
xgb_confusion <- confusionMatrix(data = predicted_levels, reference = test_data$G32)

# Print the confusion matrix to see the result
print(xgb_confusion)
