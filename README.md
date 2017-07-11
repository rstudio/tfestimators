## tfestimators - R Interface to TensorFlow Estimator API

This package provides a high-level interface to TensorFlow in R, using the [Estimator](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/python/estimator) module distributed as part of [TensorFlow](https://www.tensorflow.org/).

**IMPORTANT NOTE**: This package is under active development and the APIs are still subject to substantial changes. Help and documentation are also currently incomplete. The package is currently suitable for review and feedback but not for everyday use.

`tfestimators` is currently targeting the latest nightly builds of TensorFlow, and aims to target TensorFlow v1.3 when it is released. To install the latest nightly version of TensorFlow, you can run:

```r
tensorflow::install_tensorflow(package_url = <...>)
```

with the `package_url` argument drawn from one of the TensorFlow `.whl`s published at e.g. https://github.com/tensorflow/tensorflow#installation.

For more details and the design of the original Python API, please check out the paper: [TensorFlow Estimators: Managing Simplicity vs. Flexibility in High-Level Machine Learning Frameworks](http://terrytangyuan.github.io/data/papers/tf-estimators-kdd-paper.pdf).

To view the current documentation, clone the repository then browse the docs folder:

```r
browseURL("docs/index.html")
```

