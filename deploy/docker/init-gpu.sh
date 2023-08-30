#!/bin/bash

workers="${WORKERS:-3}"
port="${PORT:-50051}"
LOG_LEVEL=${LOG_LEVEL:-ERROR}

CUDA_MODULE_LOADING=EAGER

python3 server.py -s ${workers} -L 1 -w 0 -C /asr4_streaming_config_$LANGUAGE.toml -v "${LOG_LEVEL}" --host [::]:${port} --gpu

