#!/bin/bash

workers="${WORKERS:-3}"
port="${PORT:-50051}"
LOG_LEVEL=${LOG_LEVEL:-ERROR}

CUDA_MODULE_LOADING=EAGER
export LM="--lm-algorithm viterbi"

python3 server.py -s 1 -L ${workers} -w ${OMP_NUM_THREADS} -C /asr4_streaming_config_$LANGUAGE.toml -v "${LOG_LEVEL}" --host [::]:${port} --gpu

