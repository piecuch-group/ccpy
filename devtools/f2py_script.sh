#!/bin/bash

SOURCES=()
while [[ $# > 1 ]]
do
  key="$1"
  shift

  case "$key" in
'--source-root')
  SOURCE_ROOT="$1"
  shift
;;
'--f2py-exe')
  F2PY_EXE="$1"
  shift
;;
'--build-dir')
  BUILD_DIR="$1"
  shift
;;
*)
  SOURCES+=("$key")
;;
  esac
done


echo "Finding fortan dependencies"
ORDERED_SOURCES=$(python3 $SOURCE_ROOT/devtools/order_fortran_dependencies.py ${SOURCES[@]})
echo $ORDERED_SOURCES

echo "Building PYF"
$F2PY_EXE --quiet $ORDERED_SOURCES -m _fortran -h _fortran.pyf --overwrite-signature --build-dir $BUILD_DIR

echo "Building wrappers"
$F2PY_EXE --quiet --f2cmap $SOURCE_ROOT/devtools/f2cmap.py --build-dir $BUILD_DIR $BUILD_DIR/_fortran.pyf
