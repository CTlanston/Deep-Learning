# Models Used in the Analysis

In this analysis, three types of models were used: **Linear Regression**, a **Deep Neural Network (DNN)**, and a **Generalized Linear Model (GLM) with Lasso Regularization**. Below is a summary of each model and its purpose:

```r
# 1. Linear Regression Model
# Dataset: Boston Housing Data
# Goal: Predict the median value of owner-occupied homes (medv)

model_lm <- lm(medv~., data=Boston[train_id,])           # Fit the linear model
pred_lm <- predict(model_lm, newdata=Boston[test_id,])    # Make predictions
mean((pred_lm - Boston$medv[test_id])^2)                 # Testing error (Mean Squared Error)

# 2. Deep Neural Network (DNN)
# Dataset: Boston Housing Data (for regression) and Employee Attrition Data (for classification)
# Goal: 
#   Regression: Predict 'medv' (housing prices)
#   Classification: Predict employee attrition ('Attrition')

# DNN for Regression
model_dnn <- keras_model_sequential() %>%
  layer_dense(input_shape = 12, units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1)                          # Output layer with 1 unit for regression

# Compile the model
model_dnn %>% compile(loss = 'mse', optimizer = optimizer_rmsprop(), metrics = 'mean_absolute_error')

# Fit the model
history <- model_dnn %>% fit(x[train_id,], y[train_id], batch_size = 32, epochs = 50, validation_split = 0.2)

# DNN for Classification (Employee Attrition)
modelnn <- keras_model_sequential() %>%
  layer_dense(input_shape = ncol(x), units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(rate = 0.4) %>%
  layer_dense(units = 1, activation = 'sigmoid')    # Output layer with sigmoid for binary classification

# Compile the model
modelnn %>% compile(loss='binary_crossentropy', optimizer= optimizer_adam(), metrics = 'accuracy')

# Fit the model
history <- modelnn %>% fit(x[train_id, ], y[train_id], batch_size = 32, epochs = 20, validation_split = 0.1)

# 3. Generalized Linear Model (GLM) with Lasso Regularization
# Dataset: Employee Attrition Data
# Goal: Predict employee attrition (binary outcome) using logistic regression with Lasso regularization

# Fit the Lasso regularized GLM
cv.out <- cv.glmnet(x[train_id,], y[train_id], alpha = 1, family = 'binomial')
lambda.best <- cv.out$lambda.min
predict_lasso <- predict(cv.out, s=lambda.best, newx=x[test_id,], type='response')

# Classification accuracy with Lasso
accuracy <- function(pred, true) { mean(as.numeric(pred) == true) }
accuracy((predict_lasso > 0.5)*1, y[test_id])    # Calculate classification accuracy
