#!/bin/bash
export BATCH_SIZE=6000
#export LOC_ARR_SIZE=100 #doesnt count
#export OMP_NUM_THREADS=12
for filename in ../NEW_TRACES/*;
do
  fname=$(basename "$filename")
  export TRACE_FILE_NAME=$fname
  echo "is processing ... "$TRACE_FILE_NAME
  ./deadloc
done 
