#!/bin/bash

set -eEuo pipefail

language=$1
AWS_IP=$2
gui=$3
test=basic
interval=1

if [[ $language = @(es-es|es-mx|es-co|es-pe|es-us) ]]; then
	language="es"
fi

if [[ "$gui" == *"_upgraded"* ]]; then
  test=upgraded
fi

pip install .[client]
python bin/client.py -l "${language}" --host "${AWS_IP}" -g "${gui}" -m 
sleep 10
if [ -f "test_${language}_results.tsv" ]; then

	python tests/e2e/metrics.py --model_accuracy "test_${language}_results.tsv" \
	--expected_metrics "tests/e2e/data/expected_metrics.json" \
	--model_oov "test_${language}_oov.json" \
	--model_intratest_folder "test_${language}_intratest/" \
	--language "${language}" \
	--test_type "${test}"

	rm "test_${language}_results.tsv"
	rm "test_${language}_oov.json"
	rm -rf "test_${language}_intratest"
	rm -rf "wer"

else
	echo "There are not results for ${language} ${test} test"
	exit 1;
fi
