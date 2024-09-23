# Multivariate Time Series Forecasting with LSTM

In this code, we are working with an air pollution dataset from Beijing, China, to forecast air pollution using multivariate time series data. We are primarily using two types of Recurrent Neural Networks (RNN): **Simple RNN** and **Long Short-Term Memory (LSTM)** networks.

## Dataset Overview

The dataset includes weather and pollution data recorded each hour over a span of five years. Key variables:
- **pm2.5**: PM2.5 concentration (target variable)
- **DEWP**: Dew Point
- **TEMP**: Temperature
- **PRES**: Pressure
- **cbwd**: Combined wind direction
- **Iws**: Cumulative wind speed
- **Is**: Cumulative hours of snow
- **Ir**: Cumulative hours of rain

### Key Steps:
1. **Data Preparation**: 
    - Removed the first 24 rows due to missing values.
    - Replaced remaining missing values with zeros.
    - Split the data into training (first 4 years) and testing (last year).

2. **Feature Engineering**:
    - Created lagged versions of the time series data to predict future values based on past observations (lag-3).

3. **Modeling**:
    - We built both a **Simple RNN** and **LSTM** models to forecast `pm2.5` values using the features and lagged time series.

4. **Evaluation**:
    - Models are evaluated based on Mean Squared Error (MSE) and accuracy metrics.

---

```r
# 1. Setup and load necessary libraries
knitr::opts_chunk$set(echo = TRUE)
library(ISLR2)
library(glmnet)
library(keras)
library(readr)
library(dplyr)
use_python("C:\\Users\\ctlan\\anaconda3\\envs\\KNN")

# 2. Load dataset and handle missing data
poll <- read.csv('https://zhang-datasets.s3.us-east-2.amazonaws.com/pollution.csv')
poll <-poll[,-1]
poll <-poll[-c(1:24),]
poll[is.na(poll)] <- 0

# 3. Plot and visualize the data
plot(poll$Ir, type="l")
acf(poll$pm2.5)

# 4. Create training and testing datasets
istrain <- poll$year < 2014
xdata <- data.matrix(poll[,c(5,12)])

# 5. Function to create lagged versions of the time series
lagk <- function(x,k){
  n <- nrow(x)
  pad <- matrix(NA,k,ncol(x))
  x_lag <- rbind(pad,x[1:(n-k),])
  return(x_lag)
}

# 6. Create a lag-3 prediction frame
arframe <- data.frame(pm2.5=xdata[,1],
                      L1= lagk(xdata,1),
                      L2= lagk(xdata,2), 
                      L3= lagk(xdata,3))
arframe <-arframe[-c(1:3),]
istrain <- istrain[-c(1:3)]

# 7. Prepare the data for RNN and LSTM models
n <- nrow(arframe)
arframe <- data.matrix(arframe)
xrnn <- array(arframe[,-1], c(n,8,3))
xrnn <- xrnn[,,3:1]
xrnn <- aperm(xrnn,c(1,3,2))

dim(xrnn)

# 8. Build and train the Simple RNN model
model_rnn <- keras_model_sequential() %>%
  layer_simple_rnn(input_shape=c(3,8),units=32, dropout = 0.2) %>%
  layer_dense(units = 1) 
                    
model_rnn %>%
  compile(loss = 'mse', optimizer = optimizer_adam(), metrics = 'mean_absolute_error')

model_rnn %>%
  fit(xrnn[istrain,,], arframe[istrain,1], batch_size = 64, epochs = 50, validation_split = 0.2)

# 9. Predict and evaluate the Simple RNN model
pred_rnn <- model_rnn %>%
  predict(xrnn[!istrain,,])
mean((pred_rnn - arframe[!istrain,1])^2)

1 - mean((pred_rnn - arframe[!istrain,1])^2) / var(arframe[!istrain,1])

# 10. Build and train the LSTM model
model_lstm <- keras_model_sequential() %>%
  layer_lstm(input_shape=c(3,8), units = 32, kernel_regularizer = regularizer_l2(0.001)) %>%
  layer_dense(units = 1)

model_lstm %>%
  compile(loss='mse', optimizer = optimizer_adam(), metrics = 'mean_absolute_error')

model_lstm %>%
   fit(xrnn[istrain,,], arframe[istrain,1], batch_size=128, epochs=50, validation_split = 0.2)

# 11. Predict and evaluate the LSTM model
pred_lstm <- model_lstm() %>% predict(xrnn[!istrain,,])
1 - mean((pred_lstm - arframe[!istrain,1])^2) / var(arframe[!istrain,1])
