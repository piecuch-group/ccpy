set -e

# Install homebrew
export NONINTERACTIVE=1
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
echo 'eval "$(/usr/local/bin/brew shellenv)"' >> /Users/runner/.bash_profile
eval "$(/usr/local/bin/brew shellenv)"

env

set -x

PROJECT_DIR="$1"
PLATFORM=$(uname -m)
echo $PLATFORM

# TODO: Add licenses if need be. Also check the linux cibw script
# cat $PROJECT_DIR/tools/wheels/LICENSE_osx.txt >> $PROJECT_DIR/LICENSE.txt

# Install GFortran + OpenBLAS

if [[ $PLATFORM == "x86_64" ]]; then
  # Install openblas
  /usr/local/bin/brew update
  /usr/local/bin/brew install gcc gfortran openblas

  ln -sf  /usr/local/bin/gfortran-14 /usr/local/bin/gfortran
fi


python -m pip install -U --pre pip
python -m pip install ninja meson-python
