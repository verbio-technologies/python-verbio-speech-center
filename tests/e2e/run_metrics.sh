#!/bin/bash

language=$1
AWS_IP=$2
gui=$3
TEST_PASSED=true
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

	python tests/e2e/metrics.py --obtained_accuracy "test_${language}_results.tsv" \
	--expected_metrics "tests/e2e/data/expected_metrics.json" \
	--obtained_oov "test_${language}_oov.json" \
	--obtained_intratest "test_${language}_intratest/" \
	--language "${language}" \
	--test_type "${test}"

	rm "test_${language}_results.tsv"
	rm "test_${language}_oov.json"
	rm -rf "test_${language}_intratest"
	rm -rf "wer"

else
	echo "There are not results for ${language} ${test} test"
   	TEST_PASSED=false
fi

if [ $TEST_PASSED == false ];
then
	echo "Test did not pass"
	exit 1;
fi
