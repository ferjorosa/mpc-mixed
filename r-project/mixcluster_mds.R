library(MixCluster)

rm(list = ls())

data = read.csv("../data/mds_parkinson/mds_parkinson_train.csv")

Y = as.matrix(data)

Y[, 12:22] <- Y[, 12:22] + 1

res_2 = MixClusClustering(x=Y, g=2, burn_in=1, nbiter=1)
