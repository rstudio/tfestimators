## tfestimators - R Interface to TensorFlow Estimator API

[![Travis-CI Build Status](https://travis-ci.org/rstudio/tfestimators.svg?branch=main)](https://travis-ci.org/rstudio/tfestimators)
[![CRAN\_Status\_Badge](https://www.r-pkg.org/badges/version/tfestimators)](https://cran.r-project.org/package=tfestimators)

<img src="vignettes/images/tensorflow-architecture.png" align="right" width="50%" style="margin-left: 15px;">

The **tfestimators** package is an R interface to TensorFlow Estimators, a high-level API that provides:

- Implementations of many different model types including linear models and deep neural networks. More models are coming soon such as state saving recurrent neural networks, dynamic recurrent neural networks, support vector machines, random forest, KMeans clustering, etc.

- A flexible framework for defining arbitrary new model types as custom estimators.

- Standalone deployment of models (no R runtime required) in a wide variety of environments.

- An Experiment API that provides distributed training and hyperparameter tuning for both canned and custom estimators.

For documentation on using tfestimators, see the package website at <https://tensorflow.rstudio.com/tfestimators/>
