## this script contains functions for preprocessing
## claims data; intended to be sourced 
require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)

# Load data
load('../data/claims-raw.RData')

# function to parse html and clean text (headers + paragraphs)
parse_fn_paragraph <- function(.html){
  
  doc <- read_html(.html)
  
  if(all(is.na(doc))){
    return("")
  }
  
  raw_text <- doc %>%
    html_elements("p") %>%
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

parse_fn_header <- function(.html){
  
  doc <- read_html(.html)
  
  if(all(is.na(doc))){
    return("")
  }
  
  raw_text <- doc %>%
    # Add in header collection
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

# Preprocess data
claims_paragraph_data <- claims_raw %>%
  filter(str_detect(text_tmp, '<!')) %>%
  rowwise() %>%
  mutate(text_clean = parse_fn_paragraph(text_tmp)) %>%
  ungroup() %>%
  filter(text_clean != "") %>%
  select(.id, bclass, text_clean)

claims_header_data <- claims_raw %>%
  filter(str_detect(text_tmp, '<!')) %>%
  rowwise() %>%
  mutate(text_clean = parse_fn_header(text_tmp)) %>%
  ungroup() %>%
  filter(text_clean != "") %>%
  select(.id, bclass, text_clean)

# Setup NLP tokenizers, lemmatizer, and TF-IDF matrix calculation
nlp_fn <- function(parse_data.out){
  out <- parse_data.out %>% 
    unnest_tokens(output = token, 
                  input = text_clean, 
                  token = 'words',
                  stopwords = str_remove_all(stop_words$word, 
                                             '[[:punct:]]')) %>%
    mutate(token.lem = lemmatize_words(token)) %>%
    filter(str_length(token.lem) > 2) %>%
    count(.id, bclass, token.lem, name = 'n') %>%
    bind_tf_idf(term = token.lem, 
                document = .id,
                n = n) %>%
    pivot_wider(id_cols = c('.id', 'bclass'),
                names_from = 'token.lem',
                values_from = 'tf_idf',
                values_fill = 0)
  return(out)
}

# Get DTM data
dtm_paragraph <- nlp_fn(claims_paragraph_data)
dtm_header <- nlp_fn(claims_header_data)

# Save cleaned data for paragraph only (default training data)
output_dir <- "../data"
file_name <- "claims-cleaned.RData"
full_path <- file.path(output_dir, file_name)

# Save the objects
save(dtm_paragraph, file = full_path)

# Save cleaned data for paragraph + headers (for prelim task 1)
output_dir <- "../data"
file_name <- "claims-cleaned-headers.RData"
full_path <- file.path(output_dir, file_name)

# Save the objects
save(dtm_header, file = full_path)