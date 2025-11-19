## PRIMARY TASK: BINARY AND MULTI-CLASS CLASSIFICATION

library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)
library(tidytext)
library(textstem)
library(rvest)
library(qdapRegex)
library(stopwords)

# Load raw data
load('data/claims-raw.RData')
load('data/claims-test.RData')

# Check if mclass exists, if not we'll create it from bclass
if(!"mclass" %in% names(claims_raw)) {
  # If no mclass column exists, we might need to check the actual classes
  # For now, we'll assume bclass is the binary and we need to examine further
  print("Warning: No mclass column found. Using bclass for both models.")
  claims_raw <- claims_raw %>%
    mutate(mclass = bclass)
}

## =============================================================================
## PREPROCESSING FUNCTION (UNIFIED)
## =============================================================================

parse_fn <- function(.html){
  doc <- read_html(.html)
  
  if(all(is.na(doc))){
    return("")
  }
  
  # Extract paragraphs AND headers for better coverage
  raw_text <- doc %>%
    html_elements("p, h1, h2, h3, h4, h5, h6") %>%
    html_text2() %>%
    str_c(collapse = " ")
  
  clean_text <- raw_text %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all("'") %>%
    str_replace_all(
      paste(c('\n', '[[:punct:]]', 'nbsp', '[[:digit:]]', '[[:symbol:]]'),
            collapse = '|'),
      ' '
    ) %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
  
  return(clean_text)
}

# Preprocess training data
claims_clean <- claims_raw %>%
  filter(str_detect(text_tmp, '<!')) %>%
  rowwise() %>%
  mutate(text_clean = parse_fn(text_tmp)) %>%
  ungroup() %>%
  filter(text_clean != "") %>%
  select(.id, bclass, mclass, text_clean)

# Preprocess test data
test_clean <- claims_test %>%
  filter(str_detect(text_tmp, '<!')) %>%
  rowwise() %>%
  mutate(text_clean = parse_fn(text_tmp)) %>%
  ungroup() %>%
  filter(text_clean != "") %>%
  select(.id, text_clean)

## =============================================================================
## MODEL 1: BINARY CLASSIFICATION
## =============================================================================

set.seed(110122)

# Prepare binary classification data
train_text_binary <- claims_clean %>% pull(text_clean)
train_labels_binary <- claims_clean %>% 
  pull(bclass) %>%
  as.numeric() - 1

# Create preprocessing layer for binary model
preprocess_layer_binary <- layer_text_vectorization(
  standardize = NULL,
  split = 'whitespace',
  ngrams = NULL,
  max_tokens = 5000,  # Limit vocabulary size
  output_mode = 'tf_idf'
)

preprocess_layer_binary %>% adapt(train_text_binary)

# Define binary classification NN architecture
model_binary <- keras_model_sequential() %>%
  preprocess_layer_binary() %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(1) %>%
  layer_activation(activation = 'sigmoid')

summary(model_binary)

# Configure for training
model_binary %>% compile(
  loss = 'binary_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = 'binary_accuracy'
)

# Train binary model
history_binary <- model_binary %>%
  fit(train_text_binary, 
      train_labels_binary,
      validation_split = 0.2,
      epochs = 10,
      batch_size = 32,
      verbose = 1)

# Save binary model
save_model_tf(model_binary, "results/model-binary")

## =============================================================================
## MODEL 2: MULTI-CLASS CLASSIFICATION
## =============================================================================

# Prepare multi-class data
train_text_multi <- claims_clean %>% pull(text_clean)

# Get unique classes and create numeric labels
mclass_levels <- claims_clean %>% pull(mclass) %>% levels()
n_classes <- length(mclass_levels)

train_labels_multi <- claims_clean %>% 
  pull(mclass) %>%
  as.numeric() - 1

# One-hot encode labels for multi-class
train_labels_multi_onehot <- to_categorical(train_labels_multi, num_classes = n_classes)

# Create preprocessing layer for multi-class model
preprocess_layer_multi <- layer_text_vectorization(
  standardize = NULL,
  split = 'whitespace',
  ngrams = NULL,
  max_tokens = 5000,
  output_mode = 'tf_idf'
)

preprocess_layer_multi %>% adapt(train_text_multi)

# Define multi-class NN architecture
model_multi <- keras_model_sequential() %>%
  preprocess_layer_multi() %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 128, activation = 'relu') %>%
  layer_dropout(0.3) %>%
  layer_dense(units = 64, activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(units = 32, activation = 'relu') %>%
  layer_dropout(0.2) %>%
  layer_dense(n_classes) %>%
  layer_activation(activation = 'softmax')

summary(model_multi)

# Configure for training
model_multi %>% compile(
  loss = 'categorical_crossentropy',
  optimizer = optimizer_adam(learning_rate = 0.001),
  metrics = 'accuracy'
)

# Train multi-class model
history_multi <- model_multi %>%
  fit(train_text_multi, 
      train_labels_multi_onehot,
      validation_split = 0.2,
      epochs = 10,
      batch_size = 32,
      verbose = 1)

# Save multi-class model
save_model_tf(model_multi, "results/model-multiclass")

## =============================================================================
## GENERATE PREDICTIONS ON TEST DATA
## =============================================================================

# Binary predictions
test_text <- test_clean %>% pull(text_clean)
preds_binary <- predict(model_binary, test_text) %>% as.numeric()

bclass_levels <- claims_clean %>% pull(bclass) %>% levels()
pred_binary_classes <- factor(preds_binary > 0.5, 
                              labels = bclass_levels)

# Multi-class predictions
preds_multi <- predict(model_multi, test_text)
pred_multi_indices <- apply(preds_multi, 1, which.max) - 1
pred_multi_classes <- factor(pred_multi_indices, 
                             levels = 0:(n_classes-1),
                             labels = mclass_levels)

# Create final prediction dataframe
pred_df <- test_clean %>%
  mutate(
    bclass.pred = pred_binary_classes,
    mclass.pred = pred_multi_classes
  ) %>%
  select(.id, bclass.pred, mclass.pred)

# Save predictions (REPLACE [N] with your group number)
save(pred_df, file = 'results/preds-group2.RData')

## =============================================================================
## ESTIMATE ACCURACY (USING CROSS-VALIDATION ON TRAINING DATA)
## =============================================================================

# Create validation split
set.seed(110122)
val_split <- initial_split(claims_clean, prop = 0.8)
train_data <- training(val_split)
val_data <- testing(val_split)

# Re-train on training subset
train_text_val <- train_data %>% pull(text_clean)
train_labels_binary_val <- train_data %>% pull(bclass) %>% as.numeric() - 1
train_labels_multi_val <- train_data %>% pull(mclass) %>% as.numeric() - 1
train_labels_multi_val_onehot <- to_categorical(train_labels_multi_val, num_classes = n_classes)

# Adapt preprocessing layers on training subset
preprocess_layer_binary %>% adapt(train_text_val)
preprocess_layer_multi %>% adapt(train_text_val)

# Validate binary model
val_text <- val_data %>% pull(text_clean)
val_preds_binary <- predict(model_binary, val_text) %>% as.numeric()
val_true_binary <- val_data %>% pull(bclass)
val_pred_binary_classes <- factor(val_preds_binary > 0.5, labels = bclass_levels)

binary_accuracy <- mean(val_pred_binary_classes == val_true_binary)
cat("\nBinary Classification Accuracy:", round(binary_accuracy, 4), "\n")

# Validate multi-class model
val_preds_multi <- predict(model_multi, val_text)
val_pred_multi_indices <- apply(val_preds_multi, 1, which.max) - 1
val_true_multi <- val_data %>% pull(mclass) %>% as.numeric() - 1

multi_accuracy <- mean(val_pred_multi_indices == val_true_multi)
cat("Multi-class Classification Accuracy:", round(multi_accuracy, 4), "\n")