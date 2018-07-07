# Euclidean distance
x1 <- rnorm(30)
x2 <- rnorm(30)

Euc_dist = dist(rbind(x1, x2), method = "euclidean")
Euc_dist
# Cosine simiarlitry
vec1 = cbind( 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
vec2 = cbind( 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0 )
library('lsa')
cos(vec1,vec2) 
# Jaccard 
vec1 = c( 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0 )
vec2 = c( 0, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 0 )
library('clusterval')
cluster_similarity(vec1, vec2, similarity = "jaccard")
