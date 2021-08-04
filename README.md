# mpc-mixed
[![Build Status](https://travis-ci.com/ferjorosa/mpc-mixed.png?branch=master)](https://travis-ci.com/ferjorosa/mpc-mixed) [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

Multi-partition clustering of mixed data with Bayesian networks

## Project organization
This project is organized in several folders:

* **data**. It contains the original data and the transformed data necessary for learning the clustering models.
* **src**. Main repository of source code. It contains the Java implementations as well as the code necessary for running the experiments.
* **python-project**. Secondary repository of source code. It contains the Python source code necessary for analyzing the results of the clustering and density estimation experiments.
* **r-project**. Secondary code directory. It contains the R source code necessary for running the ClustMD[] and MixCluster[] on the Parkinson data.
* **latent_results**. Results of the density estimation experiments with 36 mixed datasets.
* **clustering_results**. Results of the clustering experiment with the Parkinson data.

## Usage
* [Project installation](https://github.com/ferjorosa/mpc-mixed/wiki)
* [Learning clustering models](https://github.com/ferjorosa/mpc-mixed/wiki/Learn-clustering-models)
* [Analysis of the results using the tool Genie](https://github.com/ferjorosa/mpc-mixed/wiki/Analysis-with-Genie)

## Disclaimer
Parkinson data should not be used for independent publications without the approval of the Movement disorders society. 

## Contact
* For any enquiries about the project, please email [ferjorosa@gmail.com](mailto:ferjorosa@gmail.com).
