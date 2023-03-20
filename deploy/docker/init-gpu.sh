#!/bin/bash

workers="${WORKERS:-3}"
port="${PORT:-50051}"
LOG_LEVEL=${LOG_LEVEL:-ERROR}

CUDA_MODULE_LOADING=EAGER
export LM="--lm-algorithm viterbi"

if [ -f asr4-${LANGUAGE}-lm.bin ]; then
    export LM="--lm-algorithm kenlm --lm-model /asr4-${LANGUAGE}-lm.bin --lexicon /asr4-${LANGUAGE}-lm.lexicon.txt"
fi

python3 server.py -s ${workers} -L 1 -w 0 -m /asr4-$LANGUAGE.onnx -d /dict.ltr.txt ${LM} -l $LANGUAGE -v "${LOG_LEVEL}" -f /format-model.$LANGUAGE.fm --host [::]:${port} --gpu

