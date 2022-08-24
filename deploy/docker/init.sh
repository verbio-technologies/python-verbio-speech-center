#!/bin/bash

set -euxo pipefail

python3 server.py -m asr4-en-us.onnx -j1