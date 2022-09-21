#!/bin/bash

set -euxo pipefail

python3 server.py -j1 -m asr4-$LANGUAGE.onnx -l $LANGUAGE -f /format-model.$LANGUAGE.fm
