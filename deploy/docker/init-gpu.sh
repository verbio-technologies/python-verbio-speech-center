#!/bin/bash

port="${PORT:-50051}"
LOG_LEVEL=${LOG_LEVEL:-ERROR}

export CUDA_MODULE_LOADING=EAGER
export W2V_GPU=${USE_GPU:="True"}
sed -i s/workers\ =\ 8/workers\ =\ 0/ /asr4_streaming_config_$LANGUAGE.toml
sed -i s/gpu\ =\ false/gpu\ =\ true/ /asr4_streaming_config_$LANGUAGE.toml

python3 server.py -s ${workers} -C /asr4_streaming_config_$LANGUAGE.toml -v "${LOG_LEVEL}" --host [::]:${port}

