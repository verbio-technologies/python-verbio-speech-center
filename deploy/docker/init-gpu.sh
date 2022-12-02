#!/bin/bash

workers="${WORKERS:-1}"
port="${PORT:-50051}"
LOG_LEVEL=${LOG_LEVEL:-ERROR}

python3 server.py -j${workers} -m /asr4-$LANGUAGE.onnx -d /dict.ltr.txt -l $LANGUAGE -v "${LOG_LEVEL}" -f /format-model.$LANGUAGE.fm --host [::]:${port} --gpu
