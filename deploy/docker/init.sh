#!/bin/bash

workers="${WORKERS:-4}"

python3 server.py -j${workers} -m /asr4-$LANGUAGE.onnx -d /dict.ltr.txt -l $LANGUAGE -f /format-model.$LANGUAGE.fm
