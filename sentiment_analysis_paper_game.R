library(sentimentr)
library(readr)
df <- read_csv('paper_game_lda.csv')
df$sentiment <- 0

# Average sentiment scores by sentences in each documents (abstract)
for (i in 1:length(df[['abstract']])){
  senti <- sentiment_by(df[[i,'abstract']])[['ave_sentiment']]
  df[[i, 'sentiment']] <- senti
  if (i %% 1000 == 0){
    print(i)
  }
}

# Divide into 3 types of sentiment (positive, neutral, negative)
df['type_of_sentiment'] <- ifelse(df['sentiment']>0.01, 'pos',ifelse(df['sentiment']< -0.01, 'neg','neu'))

library(dplyr)
df %>%
  select(X, topic_number, sentiment, type_of_sentiment) %>%
  group_by(topic_number) %>%
  summarise(count = length(X), sentiment = mean(sentiment),
            pos=sum(type_of_sentiment %in% 'pos')/count,
            neu=sum(type_of_sentiment %in% 'neu')/count,
            neg=sum(type_of_sentiment %in% 'neg')/count)
