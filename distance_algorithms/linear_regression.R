library("MASS")
data("Boston")
set.seed(0)
which_train <- sample(x = c(TRUE, FALSE), size=nrow(Boston), replace = TRUE, prob = c(0.8, 0.2))

train <- Boston[which_train, ]
test <- Boston[!which_train, ]

lm.fit =lm(medv~. ,data=train )
summary(lm.fit)
