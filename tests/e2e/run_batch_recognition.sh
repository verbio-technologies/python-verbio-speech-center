#!/bin/bash
set -eEuo pipefail

language=$1
host=$2
audio=$3
streaming=$4

if [[ $language = @(es-es|es-mx|es-co|es-pe|es-us) ]]; then
	language="es"
fi

pip install .[client,cpu]
if [ "${streaming}" == "stream" ]; then mode=""; else mode="--batch"; fi
PYTHONPATH=. python bin/client.py --no-format -c 800 -v ERROR -l "${language}" --host "${host}" -a "${audio}" ${mode} --json > "${language}-test.json"

if [ -z "$(cat ${language}-test.json | grep transcript)" ];
then
    echo "Error:"
    cat "${language}-test.json"
    exit 1;
else
    echo "Transcription is successful"
fi

exit 0
