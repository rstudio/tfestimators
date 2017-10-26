pip2.7 install --upgrade --ignore-installed --user travis pip setuptools wheel virtualenv

if [[ "$TF_VERSION" == "1.3" ]]; then
      echo "Installing TensorFlow v1.3 ...";
      Rscript -e 'tensorflow::install_tensorflow(version = "1.3")';
elif [[ "$TF_VERSION" == "1.4" ]]; then
      echo "Installing TensorFlow v1.4 ...";
      Rscript -e 'tensorflow::install_tensorflow(version = "1.4.0-rc1")';
elif [[ "$TF_VERSION" == "nightly" ]]; then
      echo "Installing TensorFlow nightly ...";
      Rscript -e 'tensorflow::install_tensorflow(version = "nightly")';
fi

mkdir inst/examples
cp -R vignettes/examples/* inst/examples
