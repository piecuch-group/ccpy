#!/bin/bash

if [ "$0" = "bash" ]; then
  SOURCE_ROOT="$BASH_SOURCE"
else
  SOURCE_ROOT="$0"
fi

PROJECT_PATH=$(realpath $(dirname "$SOURCE_ROOT"))

source $PROJECT_PATH/../ccpyvenv/bin/activate
