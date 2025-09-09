### -------------------------------------------------- ###
### ---  FINAL COMPLETE SCRIPT FOR STACKING ENSEMBLE --- ###
### -------------------------------------------------- ###

# STEP 1: SETUP THE ENVIRONMENT
# ----------------------------------------------------
library(caret)
library(caretEnsemble)
library(mlbench)
library(pROC)

# STEP 2: LOAD AND PREPARE DATA
# ----------------------------------------------------
data(PimaIndiansDiabetes2)
df <- PimaIndiansDiabetes2[complete.cases(PimaIndiansDiabetes2), ]

y <- df$diabetes
x <- df[, -which(names(df) == "diabetes")]

set.seed(42)
trainIndex <- createDataPartition(y, p = 0.8, list = FALSE)
x_train <- x[trainIndex, ]
y_train <- y[trainIndex]
x_test <- x[-trainIndex, ]
y_test <- y[-trainIndex]

# STEP 3: DEFINE AND TRAIN BASE MODELS (WITH CORRECTION)
# ----------------------------------------------------
# Define the training control method WITHOUT the problematic 'index' argument
my_control <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = "final",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Define the list of base models to train
model_list <- c("rf", "gbm", "svmRadial")

# Train the list of base models using caretList
set.seed(42)
base_models <- caretList(
  x = x_train,
  y = y_train,
  trControl = my_control,
  methodList = model_list,
  metric = "ROC"
)

# STEP 4: CREATE META-FEATURES AND TRAIN META-MODEL MANUALLY
# ----------------------------------------------------------------------
# Extract out-of-fold predictions from each base model
results <- lapply(base_models, function(model) model$pred$pos[order(model$pred$rowIndex)])

# Combine predictions into a data frame for the meta-model
meta_features_train <- as.data.frame(results)
colnames(meta_features_train) <- names(base_models)

# Define control for the meta-model
stack_control <- trainControl(
  method = "none", # No resampling needed, we already have OOF predictions
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Train the meta-model (glm) manually using caret::train
set.seed(42)
manual_stack_model <- train(
  x = meta_features_train,
  y = y_train,
  method = "glm",
  trControl = stack_control,
  metric = "ROC"
)

# Print the final manual meta-model
print(manual_stack_model)

# STEP 5: MAKE PREDICTIONS WITH THE MANUAL STACK
# ----------------------------------------------------------------------
# Get predictions from each base model on the test data
pred_rf_test <- predict(base_models$rf, newdata = x_test, type = 'prob')
pred_gbm_test <- predict(base_models$gbm, newdata = x_test, type = 'prob')
pred_svm_test <- predict(base_models$svmRadial, newdata = x_test, type = 'prob')

# Create meta-features for the test data
meta_features_test <- data.frame(
  rf = pred_rf_test$pos,
  gbm = pred_gbm_test$pos,
  svmRadial = pred_svm_test$pos
)

# Make final predictions using our manual meta-model
stack_preds_final <- predict(manual_stack_model, newdata = meta_features_test, type = "prob")

# For comparison, get predictions from the base Random Forest model
rf_preds <- predict(base_models$rf, newdata = x_test, type = "prob")


# --- Evaluation ---

# Calculate AUC for the manual stack model
stack_auc <- roc(response = y_test, predictor = stack_preds_final$pos)
print(paste("MANUAL Stack Model AUC:", round(auc(stack_auc), 4)))

# Calculate AUC for the base Random Forest model
rf_auc <- roc(response = y_test, predictor = rf_preds$pos)
print(paste("Random Forest AUC:", round(auc(rf_auc), 4)))

#Calculate AUC for the base GBM model
gbm_auc <- roc(response = y_test, predictor = pred_gbm_test$pos)
print(paste("GBM AUC:", round(auc(gbm_auc), 4)))

#Calculate AUC for the base SVM model
svm_auc <- roc(response = y_test, predictor = pred_svm_test$pos)
print(paste("SVM AUC:", round(auc(svm_auc), 4)))

