library(e1071)
data(iris)
sample = iris[sample(nrow(iris)),]
train = sample[1:105,]
test = sample[106:150,]
tune =tune(svm,Species~.,data=train,kernel ="radial",scale=FALSE,ranges
           =list(cost=c(0.001,0.01,0.1,1,5,10,100)))
tune$best.model
