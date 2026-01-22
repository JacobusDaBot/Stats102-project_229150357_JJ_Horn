library(glmnet)
library(caret)
library(mice)
library(car)
library(ggplot2)
dfb <- read.csv("Adelaide.csv")
df<-data.frame(dfb$Sunshine,dfb$MaxTemp,dfb$Humidity3pm,dfb$RainTomorrow,dfb$Evaporation)

df
# Use RainTomorrow as binary target
prepare_data_for_ridge_binary <- function(df, target_col) {
  # Make a copy and ensure target is factor
  df_clean <- df
  df_clean[[target_col]] <- as.factor(df_clean[[target_col]])
  
  # Remove the target column from features
  X <- df_clean[, !names(df_clean) %in% target_col, drop = FALSE]
  y <- df_clean[[target_col]]
  
  # Identify and remove columns with only one level
  single_level_cols <- names(X)[sapply(X, function(col) length(unique(col)) == 1)]
  if (length(single_level_cols) > 0) {
    cat("Removing columns with only one level:", paste(single_level_cols, collapse = ", "), "\n")
    X <- X[, !names(X) %in% single_level_cols, drop = FALSE]
  }
  
  # Convert to dataframe
  X_final <- as.data.frame(X)
  
  # Remove missing values
  complete_cases <- complete.cases(X_final, y)
  X_clean <- X_final[complete_cases, , drop = FALSE]
  y_clean <- y[complete_cases]
  
  # Standardize features
  preProc <- preProcess(X_clean, method = c("center", "scale"))
  X_scaled <- predict(preProc, X_clean)
  
  # Split data - now createDataPartition will work with factor
  set.seed(123)
  train_index <- createDataPartition(y_clean, p = 0.8, list = FALSE)
  
  return(list(
    X_train = X_scaled[train_index, , drop = FALSE],
    X_test = X_scaled[-train_index, , drop = FALSE],
    y_train = y_clean[train_index],
    y_test = y_clean[-train_index],
    preprocessor = preProc
  ))
}

# Use RainTomorrow as binary target
prepared_data_binary <- prepare_data_for_ridge_binary(df, "RainTomorrow")

# Fit ridge regression for binary classification
ridge_model <- cv.glmnet(
  x = as.matrix(prepared_data_binary$X_train),
  y = prepared_data_binary$y_train,
  alpha = 0,
  family = "binomial"  # For binary classification
)

print(ridge_model)
plot(ridge_model)


# Check results
print(ridge_model)
dev.new(width = 8, height = 6)  # Create a new plot window
plot(ridge_model)
# Extract coefficients for multiple lambda values
lambda_values <- exp(seq(log(0.001), log(10), length.out = 100))
coefficient_matrix <- coef(ridge_model, s = lambda_values)

# Remove the intercept row (first row)
coefficient_matrix <- coefficient_matrix[-1, ]
coefficient_matrix

# Create ridge trace plot
matplot(x = log(lambda_values), y = t(coefficient_matrix), 
        type = "l", lty = 1, lwd = 2,
        xlab = "Log(Lambda)", ylab = "Coefficient Values",
        main = "Ridge Trace Plot - All Variables",
        col = rainbow(nrow(coefficient_matrix)))

# Add vertical lines for optimal lambdas
abline(v = log(ridge_model$lambda.min), col = "red", lwd = 2, lty = 2)
abline(v = log(ridge_model$lambda.1se), col = "blue", lwd = 2, lty = 2)

# Add legend for optimal lambdas
legend("topright", 
       legend = c(paste("Lambda.min:", round(ridge_model$lambda.min, 4)),
                  paste("Lambda.1se:", round(ridge_model$lambda.1se, 4))),
       col = c("red", "blue"), lty = 2, lwd = 2)
#predictions
# Calculate VIF for ridge regression model
calculate_ridge_vif <- function(ridge_model, X_train, lambda_value = "lambda.min") {
  
  # Get the optimal lambda
  optimal_lambda <- ridge_model[[lambda_value]]
  
  # Get the covariance matrix of the features
  cov_matrix <- cov(as.matrix(X_train))
  
  # Get ridge regression coefficients (excluding intercept)
  ridge_coef <- as.numeric(coef(ridge_model, s = optimal_lambda)[-1])
  
  # Calculate the effective degrees of freedom
  # For ridge regression: df(lambda) = trace(X(X'X + lambda*I)^-1 X')
  X_matrix <- as.matrix(X_train)
  p <- ncol(X_matrix)
  I_lambda <- optimal_lambda * diag(p)
  
  # Calculate hat matrix for ridge
  hat_matrix <- X_matrix %*% solve(t(X_matrix) %*% X_matrix + I_lambda) %*% t(X_matrix)
  df_lambda <- sum(diag(hat_matrix))
  
  # Calculate VIF for each feature
  vif_values <- numeric(p)
  feature_names <- colnames(X_train)
  
  for (i in 1:p) {
    # For ridge VIF, we use the formula: VIF_j = 1 / (1 - R_j^2)
    # where R_j^2 is from regressing feature j on other features with ridge
    y <- X_matrix[, i]  # Current feature as response
    X_other <- X_matrix[, -i, drop = FALSE]  # Other features as predictors
    
    if (ncol(X_other) > 0) {
      # Fit ridge regression of feature i on all other features
      ridge_aux <- cv.glmnet(x = X_other, y = y, alpha = 0)
      y_pred <- predict(ridge_aux, newx = X_other, s = "lambda.min")
      
      # Calculate R-squared
      ss_res <- sum((y - y_pred)^2)
      ss_tot <- sum((y - mean(y))^2)
      r_squared <- 1 - (ss_res / ss_tot)
      
      # Calculate VIF
      vif_values[i] <- 1 / (1 - r_squared)
    } else {
      vif_values[i] <- 1  # Only one feature
    }
  }
  
  # Create results dataframe
  vif_results <- data.frame(
    Feature = feature_names,
    VIF = vif_values,
    Status = ifelse(vif_values > 10, "High Multicollinearity", 
                    ifelse(vif_values > 5, "Moderate Multicollinearity", "OK"))
  )
  
  # Sort by VIF (descending)
  vif_results <- vif_results[order(-vif_results$VIF), ]
  
  return(vif_results)
}

# Calculate VIF for your model
vif_results <- calculate_ridge_vif(ridge_model, prepared_data$X_train)
print(vif_results)
# Traditional VIF calculation (for OLS comparison)
calculate_ols_vif <- function(X_train) {

  
  # Create a temporary dataframe for VIF calculation
  temp_df <- as.data.frame(as.matrix(X_train))
  
  # Since we don't have an OLS model, we'll calculate VIF manually
  vif_values <- numeric(ncol(temp_df))
  feature_names <- colnames(temp_df)
  
  for (i in 1:ncol(temp_df)) {
    if (ncol(temp_df) > 1) {
      # Regress feature i on all other features
      formula_str <- paste0("`", feature_names[i], "` ~ .")
      lm_model <- lm(as.formula(formula_str), 
                     data = temp_df[, -i, drop = FALSE])
      
      # Calculate R-squared and VIF
      r_squared <- summary(lm_model)$r.squared
      vif_values[i] <- 1 / (1 - r_squared)
    } else {
      vif_values[i] <- 1
    }
  }
  
  # Create results dataframe
  vif_df <- data.frame(
    Feature = feature_names,
    VIF = vif_values,
    Status = ifelse(vif_values > 10, "High Multicollinearity", 
                    ifelse(vif_values > 5, "Moderate Multicollinearity", "OK"))
  )
  
  return(vif_df[order(-vif_df$VIF), ])
}

# Calculate traditional VIF
ols_vif <- calculate_ols_vif(prepared_data$X_train)
print("Traditional OLS VIF:")
print(ols_vif)
# Create a VIF plot
plot_vif <- function(vif_results, title = "Variance Inflation Factors") {

  
  vif_results$Feature <- factor(vif_results$Feature, 
                                levels = vif_results$Feature[order(vif_results$VIF)])
  
  ggplot(vif_results, aes(x = VIF, y = Feature, fill = Status)) +
    geom_col() +
    geom_vline(xintercept = 5, linetype = "dashed", color = "orange", alpha = 0.7) +
    geom_vline(xintercept = 10, linetype = "dashed", color = "red", alpha = 0.7) +
    scale_fill_manual(values = c("OK" = "green", 
                                 "Moderate Multicollinearity" = "orange", 
                                 "High Multicollinearity" = "red")) +
    labs(title = title,
         x = "VIF Value",
         y = "Features") +
    theme_minimal() +
    theme(legend.position = "bottom")
}

# Plot the VIF results
plot_vif(vif_results, "Ridge Regression VIF Values")
# Comprehensive VIF analysis
comprehensive_vif_analysis <- function(ridge_model, X_train) {
  cat("=== COMPREHENSIVE VIF ANALYSIS ===\n\n")
  
  # Calculate different types of VIF
  ridge_vif <- calculate_ridge_vif(ridge_model, X_train)
  ols_vif <- calculate_ols_vif(X_train)
  
  cat("1. RIDGE REGRESSION VIF:\n")
  print(ridge_vif)
  
  cat("\n2. TRADITIONAL OLS VIF (for comparison):\n")
  print(ols_vif)
  
  cat("\n3. SUMMARY STATISTICS:\n")
  cat("Ridge VIF - Mean:", round(mean(ridge_vif$VIF), 2), 
      "Max:", round(max(ridge_vif$VIF), 2), "\n")
  cat("OLS VIF - Mean:", round(mean(ols_vif$VIF), 2), 
      "Max:", round(max(ols_vif$VIF), 2), "\n")
  
  cat("\n4. MULTICOLLINEARITY ASSESSMENT:\n")
  high_vif_ridge <- sum(ridge_vif$VIF > 10)
  high_vif_ols <- sum(ols_vif$VIF > 10)
  
  cat("Features with VIF > 10 (Ridge):", high_vif_ridge, "\n")
  cat("Features with VIF > 10 (OLS):", high_vif_ols, "\n")
  
  if (high_vif_ridge == 0) {
    cat("✓ Ridge regression has successfully handled multicollinearity!\n")
  } else {
    cat("⚠ Some multicollinearity remains even after ridge regularization\n")
  }
  
  # Return both results
  return(list(ridge_vif = ridge_vif, ols_vif = ols_vif))
}


# Run comprehensive analysis
vif_analysis <- comprehensive_vif_analysis(ridge_model, prepared_data$X_train)
print(vif_analysis)

