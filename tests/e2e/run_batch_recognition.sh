#!/bin/bash
set -eEuo pipefail

language=$1
host=$2
audio=$3

if [[ $language = @(es-es|es-mx|es-co|es-pe|es-us) ]]; then
	language="es"
fi

pip install .[client]
PYTHONPATH=. python bin/client.py --no-format -v ERROR -l "${language}" --host "${host}" -a "${audio}" --batch --json > "${language}-test.json"
if [ -z "$(cat ${language}-test.json | grep transcript)" ];
then
    echo "Error:"
    cat "${language}-test.json"
    exit 1;
fi

exit 0
