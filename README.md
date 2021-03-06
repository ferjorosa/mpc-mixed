# mpc-mixed
[![Build Status](https://www.travis-ci.com/ferjorosa/mpc-mixed.svg?branch=main)](https://www.travis-ci.com/ferjorosa/mpc-mixed) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

This is the code repository of the paper [Multi-partition clustering of mixed data with Bayesian networks](https://onlinelibrary.wiley.com/doi/abs/10.1002/int.22770) (International Journal of Intelligent Systems).



## Project organization
This project is organized in several folders:

* **data**. It contains the original data and the transformed data necessary for learning the clustering models.
* **src**. Main repository of source code. It contains the Java implementations as well as the code necessary for running the experiments.
* **python-project**. Secondary repository of source code. It contains the Python source code necessary for analyzing the results of the clustering and density estimation experiments.
* **r-project**. Secondary code directory. It contains the R source code necessary for running the ClustMD and MixCluster algrithms on the Parkinson data.
* **latent_results**. Results of the density estimation experiments with 36 mixed datasets.
* **clustering_results**. Results of the clustering experiment with the Parkinson data.

## Usage
* [Project installation](https://github.com/ferjorosa/mpc-mixed/wiki)
* [Run density estimation experiments](https://github.com/ferjorosa/mpc-mixed/wiki/Density-estimation-experiments)
* [Run clustering experiment with Parkinson data](https://github.com/ferjorosa/mpc-mixed/wiki/Clustering-experiments)

## Disclaimer
Parkinson data should not be used for independent publications without the approval of the Movement disorders society. 

## Contact
* For any enquiries about the project, please email [ferjorosa@gmail.com](mailto:ferjorosa@gmail.com).
