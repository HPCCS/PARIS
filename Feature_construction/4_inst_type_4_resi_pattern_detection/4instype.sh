#!/bin/bash

for filename in ../NEW_TRACES/*;
do
  fname=$(basename "$filename")
  export TRACE_FILE_NAME=$fname
  echo "is processing ... "$TRACE_FILE_NAME
  ./4instype
done 
