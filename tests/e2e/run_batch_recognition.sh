#!/bin/bash
set -eEuo pipefail

language=$1
host=$2
audio=$3
stream=$4

if [[ $language = @(es-es|es-mx|es-co|es-pe|es-us) ]]; then
	language="es"
fi

pip install .[client]
if [ "${stream}" == "stream" ]; then
    PYTHONPATH=. python bin/client.py --no-format -v ERROR -l "${language}" --host "${host}" -a "${audio}" --json > "${language}-test.json"
else
    PYTHONPATH=. python bin/client.py --no-format -v ERROR -l "${language}" --host "${host}" -a "${audio}" --batch --json > "${language}-test.json"
fi

if [ -z "$(cat ${language}-test.json | grep transcript)" ];
then
    echo "Error:"
    cat "${language}-test.json"
    exit 1;
else
    echo "Transcription is successful"
fi

exit 0
