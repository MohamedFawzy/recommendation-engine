library(tm)
data(crude)
tdm <- TermDocumentMatrix(crude,control=list(weighting = weightTfIdf(x, normalize =TRUE), stopwords = TRUE))
inspect(tdm)