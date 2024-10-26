set -xe

NIGHTLY_FLAG=""

if [ "$#" -eq 1 ]; then
    PROJECT_DIR="$1"
elif [ "$#" -eq 2 ] && [ "$1" = "--nightly" ]; then
    NIGHTLY_FLAG="--nightly"
    PROJECT_DIR="$2"
else
    echo "Usage: $0 [--nightly] <project_dir>"
    exit 1
fi

printenv

ulimit -n 1024000
yum install -y --nogpgcheck openblas-devel.x86_64 gcc

python -m pip install ninja meson-python
