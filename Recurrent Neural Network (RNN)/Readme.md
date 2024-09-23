# Project: Recurrent Neural Networks (RNN) for Time Series Forecasting

## Task Overview
In this project, the task is to predict **log_volume** in the **NYSE** dataset using **Recurrent Neural Networks (RNN)**. The RNN model is trained to capture time dependencies and predict future values based on lagged features. We also use a linear regression model as a comparison.

## Models Used:
- **RNN (Recurrent Neural Network)**: A sequential model with a Simple RNN layer and a dense layer to predict log volume.
- **Linear Regression**: Used to compare the performance of the RNN model.

## Results:
The RNN model provides predictions of log_volume, and the mean squared error (MSE) is calculated to assess the model's performance. We also calculate the variance explained by the model using the formula \( 1 - \frac{MSE}{\text{Var}(y)} \).

### RNN Code Snippet:
```r
# Build and compile RNN model
model_rnn <- keras_model_sequential() %>%
  layer_simple_rnn(input_shape = c(5, 3), units = 16, dropout = 0.1) %>%
  layer_dense(units = 1)

model_rnn %>%
  compile(loss = 'mse', optimizer = optimizer_adam(), metrics = 'mean_absolute_error')

# Train the model
model_rnn %>%
  fit(xrnn[istrain, ], arframe$log_volume[istrain], batch_size = 32, epochs = 20, validation_split = 0.1)

