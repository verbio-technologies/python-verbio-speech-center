#!/bin/bash

set -eEuo pipefail

language=$1
AWS_IP=$2
gui=$3
expected_metrics=$4
test=basic
interval=1

if [[ $language = @(es-es|es-mx|es-co|es-pe|es-us) ]]; then
	language="es"
fi

if [[ "$gui" == *"_upgraded"* ]]; then
  test=upgraded
fi

pip install .[client]

rm "test_${language}_results.tsv" || true
rm "test_${language}_oov.json" || true
rm -rf "test_${language}_intratest" || true
rm -rf "wer" || true
rm -rf trnHypothesis.trn || true
rm -rf refHypothesis.trn || true


PYTHONPATH=. python bin/client.py -v INFO -l "${language}" --host "${AWS_IP}" -g "${gui}" -m
sleep 10
if [ -f "test_${language}_results.tsv" ]; then

	python tests/e2e/metrics.py --model_accuracy "test_${language}_results.tsv" \
	--expected_metrics "${expected_metrics}" \
	--model_oov "test_${language}_oov.json" \
	--model_intratest_folder "test_${language}_intratest/" \
	--language "${language}" \
	--test_type "${test}"

else
	echo "There are not results for ${language} ${test} test"
	exit 1;
fi
