library(randomForest)
data(iris)
sample = iris[sample(nrow(iris)),]
train = sample[1:105,]
test = sample[106:150,]
model =randomForest(Species~.,data=train,mtry=2,importance=TRUE,proximity=TRUE)
pred = predict(model,newdata=test[,-5])
