#!/bin/bash

set -euxo pipefail

export TIME=30
python bin/server.py -m /mnt/shared/squad2/projects/asr4models/asr4-en-us-0.0.15.onnx  -d /mnt/shared/squad2/projects/asr4models/asr4-en-us-0.0.15.dict.ltr.txt -j1 &
echo "Server launched, sleeping by ${TIME}"
sleep ${TIME}
