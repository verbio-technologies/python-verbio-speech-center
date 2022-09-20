#!/bin/bash

set -euxo pipefail

python3 server.py -j1 -m asr4-en-us.onnx -l en-us -f /format-model.en-us-1.0.1.fm