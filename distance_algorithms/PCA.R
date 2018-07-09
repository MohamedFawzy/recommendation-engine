data("USArrests")
head(state.name)
names(USArrests)
apply(USArrests, 2, var)

pca = prcomp(USArrests, scale= TRUE)
pca
names(pca)
pca$rotation = -pca$rotation
pca$x=pca$x
biplot(pca, scale=0)
