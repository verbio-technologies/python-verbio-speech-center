#!/bin/bash

set -eo pipefail


if [ $# -lt 3 ]
then
      echo "Usage: launch_server.sh <model_path> <dictionary_path> <formatter_path> <language> [<with_gpu>]"
      exit -1
fi

MODEL=$1
DICTIONARY=$2
FORMATTER=$3
LANGUAGE=$4

if [ -z $5 ]
then
      python3 bin/server.py -m ${MODEL} -d ${DICTIONARY} -l ${LANGUAGE} -f ${FORMATTER} -j1 &
else
      python3 bin/server.py -m ${MODEL} -d ${DICTIONARY} -l ${LANGUAGE} -f ${FORMATTER} -j1 --gpu &
fi

export TIME=30
echo "Server launched, sleeping by ${TIME}"
sleep ${TIME}
