#MF
library('recommenderlab')
data("MovieLense")
dim(MovieLense)
#applying MF using NMF
mat  = as(MovieLense,"matrix")
mat[is.na(mat)] = 0
res = nmf(mat,10)
res
#fitted values
r.hat <- fitted(res)
dim(r.hat)
p <- basis(res)
dim(p)
q <- coef(res)
dim(q)