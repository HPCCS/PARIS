#!/bin/bash
export TRACE_BACK_THRED=20
export OMP_NUM_THREADS=10
export NUM_SELF_ADD=500
export BATCH_SIZE=800
for filename in ../NEW_TRACES/*;
do
  fname=$(basename "$filename")
  export TRACE_FILE_NAME=$fname
  echo "is processing ... "$TRACE_FILE_NAME
  ./repeatadd
done 
