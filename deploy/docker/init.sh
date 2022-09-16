#!/bin/bash

set -euxo pipefail

python3 server.py -j1 -m asr4-en-us.onnx -l en-us -f /opt/verbio/data/Asr/verbio8k.en-us/formatter/en-us.fm
