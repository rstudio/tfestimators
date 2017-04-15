#!/bin/bash
set -e

cd ~/tensorflow

bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package
bazel-bin/tensorflow/tools/pip_package/build_pip_package /tmp/tensorflow_pkg

pip install --ignore-installed /tmp/tensorflow_pkg/tensorflow-1.1.0rc1-cp27-cp27m-macosx_10_12_x86_64.whl

Rscript -e 'devtools::install_github("rstudio/tensorflow")'
Rscript -e 'devtools::install_local("~/tflearn")'
