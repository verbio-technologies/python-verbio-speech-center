#!/bin/bash

port="${PORT:-50051}"
LOG_LEVEL=${LOG_LEVEL:-ERROR}

export OMP_WAIT_POLICY=PASSIVE
export KMP_AFFINITY=scatter

python3 server.py -C /asr4_streaming_config_$LANGUAGE.toml -v "${LOG_LEVEL}" --host [::]:${port}

