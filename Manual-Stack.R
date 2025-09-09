### -------------------------------------------------- ###
### ---   STACKING SCRIPT (ONLY USING 'caret')   --- ###
### -------------------------------------------------- ###

# STEP 1: SETUP THE ENVIRONMENT
# ----------------------------------------------------
# Kita tidak memuat library(caretEnsemble) lagi
library(caret)
library(mlbench)
library(pROC)

# STEP 2: LOAD AND PREPARE DATA (Sama seperti sebelumnya)
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

# STEP 3: DEFINE AND TRAIN BASE MODELS MANUALLY (Perubahan di sini)
# ----------------------------------------------------
# Definisikan training control
my_control <- trainControl(
  method = "cv",
  number = 10,
  savePredictions = "final",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Definisikan daftar model yang akan dilatih
model_list <- c("rf", "gbm", "svmRadial")

# Latih setiap model satu per satu menggunakan loop 'lapply'
# dan simpan dalam sebuah list bernama 'base_models'
set.seed(42)
base_models <- lapply(model_list, function(model) {
  print(paste("Training model:", model))
  train(
    x = x_train,
    y = y_train,
    method = model,
    trControl = my_control,
    metric = "ROC"
  )
})

# Beri nama pada setiap elemen list agar mudah diakses
names(base_models) <- model_list


# STEP 4: CREATE META-FEATURES AND TRAIN META-MODEL (Sama seperti sebelumnya)
# ----------------------------------------------------------------------
# Ekstrak prediksi "out-of-fold" dari setiap base model
results <- lapply(base_models, function(model) model$pred$pos[order(model$pred$rowIndex)])

# Gabungkan prediksi menjadi data frame
meta_features_train <- as.data.frame(results)
colnames(meta_features_train) <- names(base_models)

# Definisikan control untuk meta-model
stack_control <- trainControl(
  method = "none",
  classProbs = TRUE,
  summaryFunction = twoClassSummary
)

# Latih meta-model (glm) secara manual
set.seed(42)
manual_stack_model <- train(
  x = meta_features_train,
  y = y_train,
  method = "glm",
  trControl = stack_control,
  metric = "ROC"
)

# STEP 5: MAKE PREDICTIONS WITH THE MANUAL STACK (Sama seperti sebelumnya)
# ----------------------------------------------------------------------
# Dapatkan prediksi dari setiap base model pada data tes
pred_rf_test <- predict(base_models$rf, newdata = x_test, type = 'prob')
pred_gbm_test <- predict(base_models$gbm, newdata = x_test, type = 'prob')
pred_svm_test <- predict(base_models$svmRadial, newdata = x_test, type = 'prob')

# Buat meta-features untuk data tes
meta_features_test <- data.frame(
  rf = pred_rf_test$pos,
  gbm = pred_gbm_test$pos,
  svmRadial = pred_svm_test$pos
)

# Lakukan prediksi akhir
stack_preds_final <- predict(manual_stack_model, newdata = meta_features_test, type = "prob")

# --- Evaluation ---
# Hitung AUC
stack_auc <- roc(response = y_test, predictor = stack_preds_final$pos)
print(paste("Stack Model AUC (without caretEnsemble):", round(auc(stack_auc), 4)))