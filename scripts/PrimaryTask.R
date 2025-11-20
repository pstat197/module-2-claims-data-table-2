## PRIMARY TASK: BINARY AND MULTI-CLASS CLASSIFICATION

library(tidyverse)
library(tidymodels)
library(tidytext)
library(textstem)
library(rvest)
library(qdapRegex)
library(stopwords)
library(randomForest)
library(textrecipes)

# Load raw data
load('data/claims-raw.RData')
load('data/claims-test.RData')

# Check if mclass exists
if(!"mclass" %in% names(claims_raw)) {
  print("Warning: No mclass column found. Using bclass for both models.")
  claims_raw <- claims_raw %>%
    mutate(mclass = bclass)
}

## =============================================================================
## PREPROCESSING FUNCTION
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
cat("Preprocessing training data...\n")
claims_clean <- claims_raw %>%
  filter(str_detect(text_tmp, '<!')) %>%
  rowwise() %>%
  mutate(text_clean = parse_fn(text_tmp)) %>%
  ungroup() %>%
  filter(text_clean != "") %>%
  select(.id, bclass, mclass, text_clean)

# Preprocess test data
cat("Preprocessing test data...\n")
test_clean <- claims_test %>%
  filter(str_detect(text_tmp, '<!')) %>%
  rowwise() %>%
  mutate(text_clean = parse_fn(text_tmp)) %>%
  ungroup() %>%
  filter(text_clean != "") %>%
  select(.id, text_clean)

## =============================================================================
## CREATE TF-IDF FEATURES
## =============================================================================

cat("Creating TF-IDF features...\n")

# Tokenize and create TF-IDF for training data
tfidf_train <- claims_clean %>%
  unnest_tokens(output = token, 
                input = text_clean, 
                token = 'words') %>%
  mutate(token = lemmatize_words(token)) %>%
  filter(str_length(token) > 2) %>%
  filter(!token %in% stop_words$word) %>%
  count(.id, bclass, mclass, token) %>%
  bind_tf_idf(term = token, document = .id, n = n) %>%
  group_by(.id) %>%
  slice_max(tf_idf, n = 100) %>%  # Keep top 100 terms per document
  ungroup() %>%
  select(.id, bclass, mclass, token, tf_idf) %>%
  pivot_wider(id_cols = c(.id, bclass, mclass),
              names_from = token,
              values_from = tf_idf,
              values_fill = 0)

# Get the feature columns
feature_cols <- setdiff(names(tfidf_train), c(".id", "bclass", "mclass"))

# Tokenize and create TF-IDF for test data (using same features)
test_tokens <- test_clean %>%
  unnest_tokens(output = token, 
                input = text_clean, 
                token = 'words') %>%
  mutate(token = lemmatize_words(token)) %>%
  filter(str_length(token) > 2) %>%
  filter(!token %in% stop_words$word) %>%
  count(.id, token) %>%
  bind_tf_idf(term = token, document = .id, n = n)

# Create test matrix with same features as training
tfidf_test <- test_tokens %>%
  filter(token %in% feature_cols) %>%
  select(.id, token, tf_idf) %>%
  pivot_wider(id_cols = .id,
              names_from = token,
              values_from = tf_idf,
              values_fill = 0)

# Add missing columns with 0s
missing_cols <- setdiff(feature_cols, names(tfidf_test))
for(col in missing_cols) {
  tfidf_test[[col]] <- 0
}

# Reorder columns to match training data
tfidf_test <- tfidf_test %>%
  select(.id, all_of(feature_cols))

## =============================================================================
## MODEL 1: BINARY CLASSIFICATION (RANDOM FOREST)
## =============================================================================

cat("\n=== Training Binary Classification Model ===\n")
set.seed(110122)

# Prepare data
X_train_binary <- tfidf_train %>% select(all_of(feature_cols))
y_train_binary <- tfidf_train %>% pull(bclass)

# Train Random Forest for binary classification
cat("Training binary Random Forest...\n")
model_binary <- randomForest(
  x = X_train_binary,
  y = y_train_binary,
  ntree = 500,
  mtry = floor(sqrt(ncol(X_train_binary))),
  importance = TRUE
)

cat("Binary model trained!\n")
print(model_binary)

# Save binary model
saveRDS(model_binary, file = "results/model-binary.rds")

## MODEL 2: MULTI-CLASS CLASSIFICATION (RANDOM FOREST)

cat("\n=== Training Multi-Class Classification Model ===\n")

# Prepare data
X_train_multi <- tfidf_train %>% select(all_of(feature_cols))
y_train_multi <- tfidf_train %>% pull(mclass)

# Train Random Forest for multi-class classification
cat("Training multi-class Random Forest...\n")
model_multi <- randomForest(
  x = X_train_multi,
  y = y_train_multi,
  ntree = 500,
  mtry = floor(sqrt(ncol(X_train_multi))),
  importance = TRUE
)

cat("Multi-class model trained!\n")
print(model_multi)

# Save multi-class model
saveRDS(model_multi, file = "results/model-multiclass.rds")

## GENERATE PREDICTIONS ON TEST DATA

cat("\n=== Generating Predictions ===\n")

# Prepare test features
X_test <- tfidf_test %>% select(all_of(feature_cols))

# Binary predictions
preds_binary <- predict(model_binary, X_test)

# Multi-class predictions
preds_multi <- predict(model_multi, X_test)

# Create final prediction dataframe
pred_df <- tfidf_test %>%
  select(.id) %>%
  mutate(
    bclass.pred = preds_binary,
    mclass.pred = preds_multi
  )

# Save predictions
save(pred_df, file = 'results/preds-group2.RData')
cat("Predictions saved to results/preds-group2.RData\n")

## ESTIMATE ACCURACY (USING OUT-OF-BAG ERROR)

cat("\n=== Model Performance ===\n")

# Binary model OOB accuracy
binary_oob_error <- model_binary$err.rate[model_binary$ntree, "OOB"]
binary_accuracy <- 1 - binary_oob_error
cat("Binary Classification OOB Accuracy:", round(binary_accuracy, 4), "\n")

# Multi-class model OOB accuracy
multi_oob_error <- model_multi$err.rate[model_multi$ntree, "OOB"]
multi_accuracy <- 1 - multi_oob_error
cat("Multi-class Classification OOB Accuracy:", round(multi_accuracy, 4), "\n")

# Additional validation with a holdout set
cat("\n=== Validation Set Performance ===\n")
set.seed(110122)
train_indices <- sample(1:nrow(tfidf_train), size = 0.8 * nrow(tfidf_train))

# Binary validation
X_train_val <- X_train_binary[train_indices, ]
y_train_val <- y_train_binary[train_indices]
X_val <- X_train_binary[-train_indices, ]
y_val <- y_train_binary[-train_indices]

model_binary_val <- randomForest(
  x = X_train_val,
  y = y_train_val,
  ntree = 500,
  mtry = floor(sqrt(ncol(X_train_val)))
)

val_preds_binary <- predict(model_binary_val, X_val)
val_accuracy_binary <- mean(val_preds_binary == y_val)
cat("Binary Validation Accuracy:", round(val_accuracy_binary, 4), "\n")

# Multi-class validation
X_train_val_multi <- X_train_multi[train_indices, ]
y_train_val_multi <- y_train_multi[train_indices]
X_val_multi <- X_train_multi[-train_indices, ]
y_val_multi <- y_train_multi[-train_indices]

model_multi_val <- randomForest(
  x = X_train_val_multi,
  y = y_train_val_multi,
  ntree = 500,
  mtry = floor(sqrt(ncol(X_train_val_multi)))
)

val_preds_multi <- predict(model_multi_val, X_val_multi)
val_accuracy_multi <- mean(val_preds_multi == y_val_multi)
cat("Multi-class Validation Accuracy:", round(val_accuracy_multi, 4), "\n")

## SUMMARY

cat("\n=== SUMMARY ===\n")
cat("Binary classes:", paste(levels(y_train_binary), collapse = ", "), "\n")
cat("Multi-class classes:", paste(levels(y_train_multi), collapse = ", "), "\n")
cat("Number of training samples:", nrow(tfidf_train), "\n")
cat("Number of test samples:", nrow(tfidf_test), "\n")
cat("Number of features used:", length(feature_cols), "\n")
cat("\nModels saved to:\n")
cat("  - results/model-binary.rds\n")
cat("  - results/model-multiclass.rds\n")
cat("\nPredictions saved to:\n")
cat("  - results/preds-group2.RData\n")
cat("\nDone!\n")