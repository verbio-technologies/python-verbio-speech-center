#!/bin/bash

port="${PORT:-50051}"
LOG_LEVEL=${LOG_LEVEL:-ERROR}

export CUDA_MODULE_LOADING=EAGER
export W2V_GPU="True"

python3 server.py -C /asr4_streaming_config_$LANGUAGE.toml -v "${LOG_LEVEL}" --host [::]:${port}

