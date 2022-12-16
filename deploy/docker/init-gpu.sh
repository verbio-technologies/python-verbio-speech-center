#!/bin/bash

workers="${WORKERS:-3}"
port="${PORT:-50051}"
LOG_LEVEL=${LOG_LEVEL:-ERROR}

CUDA_MODULE_LOADING=EAGER

python3 server.py -s ${workers} -L 1 -w 0 -m /asr4-$LANGUAGE.onnx -d /dict.ltr.txt -l $LANGUAGE -v "${LOG_LEVEL}" -f /format-model.$LANGUAGE.fm --host [::]:${port} --gpu

