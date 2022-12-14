#!/bin/bash

workers="${WORKERS:-2}"
port="${PORT:-50051}"
LOG_LEVEL=${LOG_LEVEL:-ERROR}

export OMP_NUM_THREADS=8
export OMP_WAIT_POLICY=PASSIVE
export KMP_AFFINITY=scatter

python3 server.py -s 1 -L ${WORKERS} -w ${OMP_NUM_THREADS} -m /asr4-$LANGUAGE.onnx -d /dict.ltr.txt -l $LANGUAGE -v "${LOG_LEVEL}" -f /format-model.$LANGUAGE.fm --host [::]:${port}

