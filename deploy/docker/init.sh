#!/bin/bash

workers="${WORKERS:-2}"
port="${PORT:-50051}"
LOG_LEVEL=${LOG_LEVEL:-ERROR}

export OMP_NUM_THREADS=8
export OMP_WAIT_POLICY=PASSIVE
export KMP_AFFINITY=scatter
export LM="--lm-algorithm viterbi"

if [ -e asr4-${LANGUAGE}-lm.bin ]
do
    export LM="--lm-algorithm kenlm --lm-model /asr4-${LANGUAGE}-lm.bin"
done

python3 server.py -s 1 -L ${workers} -w ${OMP_NUM_THREADS} -m /asr4-$LANGUAGE.onnx -d /dict.ltr.txt ${LM} -l $LANGUAGE -v "${LOG_LEVEL}" -f /format-model.$LANGUAGE.fm --host [::]:${port}

