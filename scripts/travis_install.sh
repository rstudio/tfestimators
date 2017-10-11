pip2.7 install --upgrade --ignore-installed --user travis pip setuptools wheel virtualenv

if [[ "$TF_VERSION" == "1.3" ]]; then
      echo "Installing TensorFlow v1.3 ...";
      Rscript -e 'tensorflow::install_tensorflow(version = "1.3")';
elif [[ "$TF_VERSION" == "1.4" ]]; then
      echo "Installing TensorFlow v1.4 ...";
      Rscript -e 'tensorflow::install_tensorflow(version = "http://ci.tensorflow.org/view/Nightly/job/nightly-matrix-cpu/TF_BUILD_IS_OPT=OPT,TF_BUILD_IS_PIP=PIP,TF_BUILD_PYTHON_VERSION=PYTHON2,label=cpu-slave/lastSuccessfulBuild/artifact/pip_test/whl/tensorflow-1.head-cp27-cp27mu-linux_x86_64.whl")';
fi

mkdir inst/examples
cp -R vignettes/examples/* inst/examples
