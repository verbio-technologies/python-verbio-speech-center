#!/bin/bash

set -euxo pipefail

python3 server.py -j1 -m /asr4-$LANGUAGE.onnx -d /dict.ltr.txt -l $LANGUAGE -f /format-model.$LANGUAGE.fm
