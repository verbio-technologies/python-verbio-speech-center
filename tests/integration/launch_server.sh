#!/bin/bash

set -eo pipefail


if [ $# -lt 3 ]
then
      echo "Usage: launch_server.sh <model_path> <dictionary_path> <language> [<with_gpu>]"
      exit -1
fi

MODEL=$1
DICTIONARY=$2
LANGUAGE=$3

if [ -z $4 ]
then
      python bin/server.py -m ${MODEL} -d ${DICTIONARY} -l ${LANGUAGE} -j1 &
else
      python bin/server.py -m ${MODEL} -d ${DICTIONARY} -l ${LANGUAGE} -j1 --gpu &
fi

export TIME=30
echo "Server launched, sleeping by ${TIME}"
sleep ${TIME}
