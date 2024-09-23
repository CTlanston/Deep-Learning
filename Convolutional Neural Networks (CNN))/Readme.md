# Project: Convolutional Neural Networks (CNN) for Image Classification

## Overview
This project focuses on using Convolutional Neural Networks (CNNs) for image classification tasks, specifically the **Fashion MNIST** dataset. CNNs are particularly effective for image recognition due to their ability to capture spatial hierarchies in data by applying convolutional filters and pooling layers. We implement a CNN with two convolutional layers followed by fully connected layers to classify fashion images into 10 categories (e.g., T-shirts, trousers, coats, etc.).

## Models Used
- **Convolutional Layers**: Extract features from the image using filters.
- **Max Pooling Layers**: Reduce the spatial dimensions of the feature maps.
- **Dropout Layer**: Prevent overfitting by randomly ignoring certain neurons during training.
- **Fully Connected Dense Layers**: Classify the extracted features into 10 categories using softmax activation.

## Results
After training for 10 epochs, the CNN achieved approximately **97% accuracy** on the test dataset, demonstrating the effectiveness of CNNs for image classification tasks.

### CNN Code Snippet:
```r
# Build the CNN model
model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), padding = "same", activation = "relu", input_shape = c(28, 28, 1)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), padding = "same", activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 512, activation = "relu") %>%
  layer_dense(units = 10, activation = "softmax")

# Compile the model
model %>% compile(loss = "categorical_crossentropy", optimizer = optimizer_rmsprop(), metrics = c("accuracy"))

# Train the model
history <- model %>% fit(x_train, y_train, epochs = 10, batch_size = 128, validation_split = 0.2)

# Evaluate the model
model %>% evaluate(x_test, y_test)
