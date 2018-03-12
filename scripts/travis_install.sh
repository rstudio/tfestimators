pip2.7 install --upgrade --ignore-installed --user travis pip setuptools wheel virtualenv

echo "Installing TensorFlow v$TF_VERSION..."
Rscript -e 'tensorflow::install_tensorflow(version = Sys.getenv("TF_VERSION"))';

mkdir inst/examples
cp -R vignettes/examples/* inst/examples
