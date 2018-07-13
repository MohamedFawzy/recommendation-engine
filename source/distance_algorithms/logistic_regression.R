set.seed(1)
x1 = rnorm(1000) # sample continuous variables
x2 = rnorm(1000)
z = 1 + 4*x1 + 3*x2 # data creation
pr = 1/(1+exp(-z)) # applying logit function
y = rbinom(1000,1,pr) # bernoulli response variable

#now feed it to glm:
df = data.frame(y=y,x1=x1,x2=x2)
glm( y~x1+x2,data=df,family="binomial")

  