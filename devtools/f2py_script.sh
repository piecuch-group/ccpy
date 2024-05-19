#!/bin/bash


SOURCES=()
while [[ $# > 1 ]]
do
  key="$1"
  shift

  case "$key" in
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


echo "Building PYF"
$F2PY_EXE ${SOURCES[@]} -m optimizations -h optimizations.pyf --build-dir $BUILD_DIR

echo "Building wrappers"
$F2PY_EXE $BUILD_DIR/optimizations.pyf --build-dir $BUILD_DIR
