# Script for generating the ClustMD model. We establish a limit on the number of 
# iterations and clusters because we have previously explored the model with
# several clusters and we have also explored the method without establishing any
# limits. 

# In these tests, we observed a bug that deteriorated the quality of the results
# (hence why we stop one iteration before it happens) and also observed that the
# best model  had 3 clusters (according to BIC). You can test it yourself, by
# executing the commented sections of code.

################################################################################

# Problem with BIC in clustMD:

# ClustMD follows the definition of BIC score that is presented in Wikipedia
# (https://en.wikipedia.org/wiki/Bayesian_information_criterion). This formula is
# slightly different from the one that is used in the context of Bayesian networks.
# For this reason, we are going to manually estimate the BIC score using the one
# from BNs, which is also considered by other R packages such as mixCluster (also
# present in this article). The formula is the following:
#
# BIC = LL -dim(M)*log(N)/2

# In order to estimate the proper BIC score of the resulting models, we need to
# know the number of parameters. However, clustMD doesn't offer a public method
# or attribute to know the number of parameters. However, it does offer a private
# method (npars_clustMD) to estimate the number of parameters For this reason, we 
# modify the current version of clustMD via "trace" and add the following lines 
# of code below line 291.

# trace(clustMD::clustMD, edit = TRUE)

##### Code to inject below line 291:

# num_params <- npars_clustMD(model, D, G, J, CnsIndx, OrdIndx, K)
# print(paste("Number of parameters: ", num_params))

################################################################################

library(clustMD)

rm(list=ls())

data = read.csv("../data/mds_parkinson/mds_parkinson_train.csv")

Y = as.matrix(data)

Y[, 12:22] <- Y[, 12:22] + 1

################## Result with 2 clusters
res_2 <- clustMD::clustMD(X = Y, G = 2, CnsIndx = 11, OrdIndx = 22, Nnorms = 20000,
                          MaxIter = 8, model = "EVI", store.params = TRUE, scale = FALSE,
                          startCL = "kmeans", autoStop= TRUE, ma.band=30, stop.tol=0.0001)

num_params_2 <- 88
ll_2 <- res_2$likelihood.store[8]
bic_2 <- ll_2 - (num_params_2 * log(nrow(data)) / 2)
print(paste("Log-likelihood score with 2 clusters: ", ll_2))
print(paste("BIC score with 2 clusters: ", bic_2))
print(paste("Wikipedia BIC score with 2 clusters: ", res_2$BIChat))

################## Result with 3 clusters
res_3 <- clustMD::clustMD(X = Y, G = 3, CnsIndx = 11, OrdIndx = 22, Nnorms = 20000,
                          MaxIter = 6, model = "EVI", store.params = TRUE, scale = FALSE,
                          startCL = "kmeans", autoStop= TRUE, ma.band=30, stop.tol=0.0001)

num_params_3 <- 132
ll_3 <- res_3$likelihood.store[6]
bic_3 <- ll_3 - (num_params_3 * log(nrow(data)) / 2)
print(paste("Log-likelihood score with 3 clusters: ", ll_3))
print(paste("BIC score with 3 clusters: ", bic_3))
print(paste("Wikipedia BIC score with 3 clusters: ", res_3$BIChat))

################## Result with 4 clusters
res_4 <- clustMD::clustMD(X = Y, G = 4, CnsIndx = 11, OrdIndx = 22, Nnorms = 20000,
                          MaxIter = 4, model = "EVI", store.params = TRUE, scale = FALSE,
                          startCL = "kmeans", autoStop= TRUE, ma.band=30, stop.tol=0.0001)

num_params_4 <- 176
ll_4 <- res_4$likelihood.store[4]
bic_4 <- ll_4 - (num_params_4 * log(nrow(data)) / 2)
print(paste("Log-likelihood score with 4 clusters: ", ll_4))
print(paste("My BIC score with 4 clusters: ", bic_4))
print(paste("Wikipedia BIC score with 4 clusters: ", res_4$BIChat))

################## Result with 5 clusters
res_5 <- clustMD::clustMD(X = Y, G = 5, CnsIndx = 11, OrdIndx = 22, Nnorms = 20000,
                          MaxIter = 4, model = "EVI", store.params = TRUE, scale = FALSE,
                          startCL = "kmeans", autoStop= TRUE, ma.band=30, stop.tol=0.0001)

num_params_5 <- 220
ll_5 <- res_5$likelihood.store[4]
bic_5 <- ll_5 - (num_params_5 * log(nrow(data)) / 2)
print(paste("Log-likelihood score with 5 clusters: ", ll_5))
print(paste("My BIC score with 5 clusters: ", bic_5))
print(paste("Wikipedia BIC score with 5 clusters: ", res_5$BIChat))