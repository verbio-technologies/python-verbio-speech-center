#!/bin/bash

workers="${WORKERS:-default_value}"

python3 server.py -j${workers} -m /asr4-$LANGUAGE.onnx -d /dict.ltr.txt -l $LANGUAGE -f /format-model.$LANGUAGE.fm
