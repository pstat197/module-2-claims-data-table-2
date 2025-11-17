## this script contains functions for preprocessing
## claims data; intended to be sourced 
require(tidyverse)
require(tidytext)
require(textstem)
require(rvest)
require(qdapRegex)
require(stopwords)
require(tokenizers)

# function to parse html and clean text (headers + paragraphs)
parse_fn <- function(.html, include_headers){
  
  doc <- read_html(.html)
  
  # maintain paragraph scraping functionality
  paragraphs <- doc %>%
    html_elements('p') %>%
    html_text2()
  
  if (include_headers == True){
    # add in header extraction
    headers <- doc %>%
      html_elements('h1, h2, h3, h4, h5, h6') %>%
      html_text2()
    raw_text <- c(headers, paragraphs) %>%
      str_c(collapse = " ")
  } else {
    raw_text <- paragraphs %>%
      str_c(collapse = " ")
  }
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

# function to apply to claims data
parse_data <- function(.df){
  out <- .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn(text_tmp)) %>%
    unnest(text_clean) 
  return(out)
}

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