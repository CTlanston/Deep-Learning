# Linear Regression with Regularization (Ridge and Lasso)

## Overview

Regularization techniques such as Ridge and Lasso regression are used to improve the generalization of a linear regression model by adding penalties for large coefficients.

- **Ridge Regression** adds an L2 penalty (the squared magnitude of coefficients).
- **Lasso Regression** adds an L1 penalty (the absolute value of coefficients), which can result in sparse models with some coefficients exactly equal to zero.

### Key Concepts:

1. **Ridge Regression**:
    - Minimizes the residual sum of squares (RSS) plus the squared value of the coefficients.
    - Suitable when all predictors have some effect on the response variable.
    
    Ridge Loss Function:
    \[
    \text{RSS} + \lambda \sum_{i=1}^{p} \beta_i^2
    \]

2. **Lasso Regression**:
    - Minimizes the RSS plus the absolute value of the coefficients.
    - Encourages sparsity, potentially driving some coefficients to zero, which helps in feature selection.
    
    Lasso Loss Function:
    \[
    \text{RSS} + \lambda \sum_{i=1}^{p} |\beta_i|
    \]

### R Code

1. **Load the required libraries**:
    We use the `glmnet` package to implement Ridge and Lasso regression.
    ```r
    library(glmnet)
    ```

2. **Prepare the data**:
    Split the data into a matrix of predictors `X` and a response variable `y`.
    ```r
    X <- as.matrix(your_dataframe[, -1])  # Predictor matrix
    y <- your_dataframe$response_variable  # Response variable
    ```

3. **Ridge Regression**:
    Use the `glmnet` function with `alpha = 0` for Ridge regression.
    ```r
    ridge_model <- glmnet(X, y, alpha = 0)
    ```

4. **Lasso Regression**:
    Use the `glmnet` function with `alpha = 1` for Lasso regression.
    ```r
    lasso_model <- glmnet(X, y, alpha = 1)
    ```

5. **Cross-Validation**:
    Use cross-validation to find the optimal lambda (penalty) parameter for both models.
    ```r
    cv_ridge <- cv.glmnet(X, y, alpha = 0)
    cv_lasso <- cv.glmnet(X, y, alpha = 1)
    ```

6. **Get Optimal Lambda**:
    Extract the lambda that minimizes the cross-validated error.
    ```r
    best_lambda_ridge <- cv_ridge$lambda.min
    best_lambda_lasso <- cv_lasso$lambda.min
    ```

7. **Final Models**:
    Fit the final models using the optimal lambda.
    ```r
    final_ridge <- glmnet(X, y, alpha = 0, lambda = best_lambda_ridge)
    final_lasso <- glmnet(X, y, alpha = 1, lambda = best_lambda_lasso)
    ```

### Conclusion

By using Ridge and Lasso regression, we can balance between underfitting and overfitting by penalizing large coefficients. Ridge
