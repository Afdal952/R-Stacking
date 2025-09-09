### -------------------------------------------------- ###
### ---  FINAL SCRIPT WITH HYPERPARAMETER TUNING   --- ###
### -------------------------------------------------- ###

# STEP 1: SETUP THE ENVIRONMENT
# ----------------------------------------------------
library(caret)
# library(caretEnsemble) # Tidak kita gunakan lagi di versi ini
library(mlbench)
library(pROC)

# STEP 2: LOAD AND PREPARE DATA (Sama seperti sebelumnya)
# ----------------------------------------------------
data(PimaIndiansDiabetes2)
df <- PimaIndiansDiabetes2[complete.cases(PimaIndiansDiabetes2), ]

y <- df$diabetes
x <- df[, -which(names(df) == "diabetes")]

set.seed(15)
trainIndex <- createDataPartition(y, p = 0.7, list = FALSE)
x_train <- x[trainIndex, ]
y_train <- y[trainIndex]
x_test <- x[-trainIndex, ]
y_test <- y[-trainIndex]

# STEP 3: DEFINE AND TUNE BASE MODELS (MODIFIED)
# ----------------------------------------------------
# Definisikan training control, ini akan kita gunakan untuk semua model
my_control <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = "final",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# --- Proses Tuning Dimulai di Sini ---
set.seed(15)

# 1. Tune Random Forest (rf)
print("Tuning Random Forest...")
# Membuat daftar hyperparameter 'mtry' yang ingin diuji
rf_grid <- expand.grid(mtry = c(2, 4, 6, 8)) 
base_model_rf <- train(
  x = x_train, y = y_train,
  method = "rf",
  trControl = my_control,
  metric = "ROC",
  tuneGrid = rf_grid
)
print("Hasil Tuning RF Terbaik:")
print(base_model_rf$bestTune)


# 2. Tune Gradient Boosting Machine (gbm)
print("Tuning GBM...")
# Membuat grid yang lebih kompleks untuk GBM
gbm_grid <- expand.grid(
  interaction.depth = c(1, 3, 5),
  n.trees = (1:5) * 50, # Menguji 50, 100, 150, 200, 250 pohon
  shrinkage = c(0.01, 0.1),
  n.minobsinnode = c(10, 20)
)
base_model_gbm <- train(
  x = x_train, y = y_train,
  method = "gbm",
  trControl = my_control,
  metric = "ROC",
  tuneGrid = gbm_grid,
  verbose = FALSE # Mematikan log training gbm yang panjang
)
print("Hasil Tuning GBM Terbaik:")
print(base_model_gbm$bestTune)


# 3. Tune Support Vector Machine (svmRadial)
print("Tuning SVM...")
# Membuat grid untuk parameter C (cost) dan sigma
svm_grid <- expand.grid(
  sigma = c(0.01, 0.05, 0.1, 0.5),
  C = c(0.75, 1, 1.25, 1.5)
)
base_model_svm <- train(
  x = x_train, y = y_train,
  method = "svmRadial",
  trControl = my_control,
  metric = "ROC",
  tuneGrid = svm_grid
)
print("Hasil Tuning SVM Terbaik:")
print(base_model_svm$bestTune)


# Gabungkan model-model yang SUDAH DI-TUNING ke dalam satu list
base_models <- list(
  rf = base_model_rf,
  gbm = base_model_gbm,
  svmRadial = base_model_svm
)

# STEP 4: CREATE META-FEATURES AND TRAIN META-MODEL (Sama seperti sebelumnya)
# ----------------------------------------------------------------------
# (Tidak ada perubahan di sini, langkah ini menggunakan 'base_models' yang sudah optimal)
results <- lapply(base_models, function(model) model$pred$pos[order(model$pred$rowIndex)])
meta_features_train <- as.data.frame(results)
colnames(meta_features_train) <- names(base_models)

stack_control <- trainControl(
  method = "none",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

set.seed(15)
manual_stack_model <- train(
  x = meta_features_train,
  y = y_train,
  method = "glm",
  trControl = stack_control,
  metric = "ROC"
)

# STEP 5 & Evaluation (Sama seperti sebelumnya)
# ----------------------------------------------------------------------
# (Tidak ada perubahan di sini, langkah ini menggunakan 'base_models' yang sudah optimal)
pred_rf_test <- predict(base_models$rf, newdata = x_test, type = 'prob')
pred_gbm_test <- predict(base_models$gbm, newdata = x_test, type = 'prob')
pred_svm_test <- predict(base_models$svmRadial, newdata = x_test, type = 'prob')

meta_features_test <- data.frame(
  rf = pred_rf_test$pos,
  gbm = pred_gbm_test$pos,
  svmRadial = pred_svm_test$pos
)

stack_preds_final <- predict(manual_stack_model, newdata = meta_features_test, type = "prob")

# --- Evaluation ---
# Hitung dan bandingkan semua AUC
stack_auc <- roc(response = y_test, predictor = stack_preds_final$pos)
rf_auc_tuned <- roc(response = y_test, predictor = pred_rf_test$pos)
gbm_auc_tuned <- roc(response = y_test, predictor = pred_gbm_test$pos)
svm_auc_tuned <- roc(response = y_test, predictor = pred_svm_test$pos)

print(paste("TUNED Stack Model AUC:", round(auc(stack_auc), 4)))
print(paste("TUNED RF Model AUC:", round(auc(rf_auc_tuned), 4)))
print(paste("TUNED GBM Model AUC:", round(auc(gbm_auc_tuned), 4)))
print(paste("TUNED SVM Model AUC:", round(auc(svm_auc_tuned), 4)))