#!/bin/bash

set -eo pipefail

if [ $# -lt 3 ]
then
      echo "Usage: launch_server.sh <model_path> <dictionary_path> <formatter_path> <language> [<with_gpu>]"
      exit -1
fi

MODEL=$1
DICTIONARY=$2
LANGUAGE=$4
FORMATTER=$(ls $3/format-model.${LANGUAGE}*)
export CUDA_VISIBLE_DEVICES=1

if [ -z $5 ]
then
      python3 bin/server.py -m ${MODEL} -d ${DICTIONARY} -l ${LANGUAGE} -f ${FORMATTER} -s1 -L1 -w2 -v TRACE &
else
      python3 bin/server.py -m ${MODEL} -d ${DICTIONARY} -l ${LANGUAGE} -f ${FORMATTER} -s1 -L1 -w2 --gpu -v TRACE &
fi

export TIME=30
echo "Server launched, sleeping by ${TIME}"
sleep ${TIME}
