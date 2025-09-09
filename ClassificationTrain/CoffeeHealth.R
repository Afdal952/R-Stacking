#Perform Some Classification Algorithms on CoffeeHealth DataSet
#1. LOAD THE LIBRARIES####
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
library(gbm)

#2. LOAD THE DATA####
#Open the working directory on D:/Projects-R
setwd("D:/Projects-R")
coffee <- read.csv("Data/CoffeeHealth/synthetic_coffee.csv")
View(coffee)

#3. DATA PREPROCESSING####
#Check for missing values
sum(is.na(coffee))

#Check the structure of the data
str(coffee)

#Convert all categorical variables to factors
categorical_vars <- c("Gender", "Sleep_Quality", "Stress_Level", "Health_Issues", "Occupation") 
coffee[categorical_vars] <- lapply(coffee[categorical_vars], as.factor)

str(coffee)

#Split data into train and test#
set.seed(123) # For reproducibility
train_index <- sample(1:nrow(coffee), 0.75 * nrow(coffee))
train_data <- coffee[train_index, ]
test_data <- coffee[-train_index, ]

#4. APPLY CLASSIFICATION ALGORITHMS####
####4.1. Multinomial Logistic Regression####
logreg <- multinom(Health_Issues~., data = train_data, maxit = 250, trace = FALSE, tol = 1e-5)
logreg_pred <- predict(logreg, newdata = test_data)
confusionMatrix(logreg_pred, test_data$Health_Issues)

####4.2. Decision Tree####
dtree <- rpart(Health_Issues~., data = train_data, method = "class")
dtree_pred <- predict(dtree, newdata = test_data, type = "class")
confusionMatrix(dtree_pred, test_data$Health_Issues)

####4.3. Random Forest####
rf <- randomForest(Health_Issues~., data = train_data, ntree = 150, mtry = 5)
rf_pred <- predict(rf, newdata = test_data)
confusionMatrix(rf_pred, test_data$Health_Issues)

####4.4. Support Vector Machine####
SVM <- svm(Health_Issues~., data = train_data, kernel = "radial")
SVM_pred <- predict(SVM, newdata = test_data)
confusionMatrix(SVM_pred, test_data$Health_Issues)

####4.5. Naive Bayes####
nb <- naive_bayes(Health_Issues~., data = train_data, )
nb_pred <- predict(nb, newdata = test_data)
confusionMatrix(nb_pred, test_data$Health_Issues)

####4.6. Neural Networks####
# Salin data & simpan label target (dimulai dari 0)
train_label <- as.numeric(train_data$Health_Issues) - 1
test_label <- as.numeric(test_data$Health_Issues) - 1

# Siapkan data fitur dengan menghapus kolom target dan ID
train_features <- train_data[, !(names(train_data) %in% c("ID", "Health_Issues"))]
test_features <- test_data[, !(names(test_data) %in% c("ID", "Health_Issues"))]

# Lakukan one-hot encoding pada fitur
dmy <- dummyVars(" ~ .", data = train_features)
train_matrix <- predict(dmy, newdata = train_features)
test_matrix <- predict(dmy, newdata = test_features)
# Buat model scaling (center & scale) dari data training
scaler <- preProcess(train_matrix, method = c("center", "scale"))

# Terapkan scaling ke data training dan testing
train_scaled <- predict(scaler, train_matrix)
test_scaled <- predict(scaler, test_matrix)

# Gabungkan data training yang sudah di-scaling dengan targetnya
train_data_nn <- as.data.frame(train_scaled)
train_data_nn$Health_Issues <- train_data$Health_Issues

# Latih model Neural Network
set.seed(123)
nn_fixed <- nnet(
  Health_Issues ~ ., 
  data = train_data_nn, 
  size = 10,
  maxit = 200,
  decay = 5e-4,
  trace = FALSE
)

# Lakukan prediksi pada data testing yang sudah di-scaling
nn_pred_fixed <- predict(nn_fixed, newdata = as.data.frame(test_scaled), type = "class")

# Konversi hasil prediksi ke format factor
nn_pred_fixed <- as.factor(nn_pred_fixed)

# Tampilkan confusion matrix
confusionMatrix(nn_pred_fixed, test_data$Health_Issues)

####4.7. K-Nearest Neighbors####
KNN <- train(
  Health_Issues ~ ., 
  data = train_data, 
  method = "knn", 
  tuneLength = 20,
  trControl = trainControl(method = "cv", number = 5)
)

KNN_pred <- predict(KNN, newdata = test_data)
confusionMatrix(KNN_pred, test_data$Health_Issues)

####4.8. XGBoost####
# Buat DMatrix untuk XGBoost
dtrain <- xgb.DMatrix(data = as.matrix(train_matrix), label = train_label)
dtest <- xgb.DMatrix(data = as.matrix(test_matrix), label = test_label)

# Tentukan parameter model
num_class <- length(levels(train_data$Health_Issues))
params <- list(
  objective = "multi:softmax",
  eta = 0.1,
  max_depth = 4,
  num_class = num_class
)

# Latih model XGBoost
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 100)

# Lakukan prediksi pada data test
xgb_pred_numeric <- predict(xgb_model, dtest)

# Konversi hasil prediksi kembali ke nama kelas asli
class_levels <- levels(test_data$Health_Issues)
xgb_pred <- factor(class_levels[xgb_pred_numeric + 1], levels = class_levels)

# Tampilkan confusion matrix
confusionMatrix(xgb_pred, test_data$Health_Issues)

#5. SETOR HASIL AKURASI MASING2 ALGORITMA####
# Buat data frame untuk menyimpan hasil akurasi
results <- data.frame(
  Algorithm = c("Logistic Regression", "Decision Tree", "Random Forest", 
                "Support Vector Machine", "Naive Bayes", "Neural Networks", 
                "K-Nearest Neighbors", "XGBoost"),
  Accuracy = c(
    confusionMatrix(logreg_pred, test_data$Health_Issues)$overall['Accuracy'],
    confusionMatrix(dtree_pred, test_data$Health_Issues)$overall['Accuracy'],
    confusionMatrix(rf_pred, test_data$Health_Issues)$overall['Accuracy'],
    confusionMatrix(SVM_pred, test_data$Health_Issues)$overall['Accuracy'],
    confusionMatrix(nb_pred, test_data$Health_Issues)$overall['Accuracy'],
    confusionMatrix(nn_pred_fixed, test_data$Health_Issues)$overall['Accuracy'],
    confusionMatrix(KNN_pred, test_data$Health_Issues)$overall['Accuracy'],
    confusionMatrix(xgb_pred, test_data$Health_Issues)$overall['Accuracy']
  )
)
print(results)

#6. STACKING ENSEMBLE METHOD (REVISED)####

# --- LANGKAH 1: PERSIAPAN DATA ---
# Pisahkan fitur dan target dari data training & testing
train_features <- train_data[, !(names(train_data) %in% c("ID", "Health_Issues"))]
train_target <- train_data$Health_Issues
test_features <- test_data[, !(names(test_data) %in% c("ID", "Health_Issues"))]
test_target <- test_data$Health_Issues


# --- LANGKAH 2: PREPROCESSING PIPELINE ---
# Buat resep one-hot encoding dan scaling dari data training
dmy_recipe <- dummyVars(" ~ .", data = train_features)
scaler_recipe <- preProcess(predict(dmy_recipe, train_features), method = c("center", "scale"))

# Terapkan resep ke data training
train_matrix <- predict(dmy_recipe, newdata = train_features)
train_scaled <- predict(scaler_recipe, train_matrix)
train_final <- as.data.frame(train_scaled)
train_final$Health_Issues <- train_target

# Terapkan resep ke data testing dan buat `test_final`
test_matrix <- predict(dmy_recipe, newdata = test_features)
test_scaled <- predict(scaler_recipe, test_matrix)
test_final <- as.data.frame(test_scaled) # <-- Variabel yang hilang ditambahkan di sini


# --- LANGKAH 3: LATIH BASE MODELS ---
# Tentukan dan latih beberapa model dasar
BaseModels <- list(
  rf = caretModelSpec(method = "rf", tuneLength = 3),
  xgb = caretModelSpec(method = "xgbTree", tuneLength = 3),
  nn = caretModelSpec(method = "nnet", tuneLength = 3, trace = FALSE, maxit = 200)
)

suppressMessages({
  set.seed(123)
  stacked_models <- suppressWarnings(caretList(
    Health_Issues ~ .,
    data = train_final,
    trControl = trainControl(method = "cv", number = 5, savePredictions = "final"),
    tuneList = BaseModels
  ))
})


# --- LANGKAH 4: LATIH META-MODEL ---
# Buat data training untuk meta-model dari prediksi base models
meta_train_data <- data.frame(
  rf = stacked_models$rf$pred$pred,
  xgb = stacked_models$xgb$pred$pred,
  nn = stacked_models$nn$pred$pred,
  Health_Issues = stacked_models$rf$pred$obs
)

# Bersihkan data dari NA dan latih meta-model
meta_train_data_clean <- na.omit(meta_train_data)
stackControl <- trainControl(method = "cv", number = 5)
set.seed(456)
meta_model_final <- train(
  Health_Issues ~ .,
  data = meta_train_data_clean,
  method = "gbm",
  trControl = stackControl,
  tuneLength = 3,
  verbose = FALSE
)
print(meta_model_final)


# --- LANGKAH 5: PREDIKSI DAN EVALUASI ---
# Buat data test untuk meta-model
meta_test_data <- data.frame(
  rf = predict(stacked_models$rf, newdata = test_final),
  xgb = predict(stacked_models$xgb, newdata = test_final),
  nn = predict(stacked_models$nn, newdata = test_final)
)

# Lakukan prediksi akhir dan evaluasi
final_predictions <- predict(meta_model_final, newdata = meta_test_data)
final_predictions_factored <- factor(final_predictions, levels = levels(test_target))
confusionMatrix(final_predictions_factored, test_target)


