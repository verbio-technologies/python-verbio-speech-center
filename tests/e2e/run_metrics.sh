#!/bin/bash

language=$1
AWS_IP=$2
gui=$3
TEST_PASSED=true
variant="none"


if [[ $language = @(es-es|es-mx|es-co|es-pe|es-us) ]]; then
	variant=$(echo $language | cut -d "-" -f 2)
	language="es"
fi

function compare_metrics(){
	metric="$1"
	expected_metric="$2"

	if (( $(echo "$metric == $expected_metric" |bc -l) ));
	then
	  echo "Obtained values and expected values match"
	elif (( $(echo "$metric > $expected_metric" |bc -l) ));
	then
		echo "Obtained value (${metric}) is higher than expected value (${expected_metric})"
		TEST_PASSED=false
	else 
		echo "Obtained value (${metric}) is lower than expected value (${expected_metric})"
		TEST_PASSED=false
	fi
}


pip install .[client]
python bin/client.py -l "${language}" --host "${AWS_IP}:50051" -g "${gui}" -m
sleep 10
wer_metric=$(cat "test_${language}_results.tsv" | grep "Accuracy" | cut -d " " -f 2)
oov_metric=$(jq '.score' "test_${language}_oov.json" | sed 's/[^0-9.]*//g')


if [ "${variant}" != "none" ];
then

	expected_wer=$(jq --arg keyvar "$language-$variant" '.[$keyvar].accuracy' "tests/e2e/data/expected_metrics.json")
	expected_oov=$(jq --arg keyvar "$language-$variant" '.[$keyvar].oov' "tests/e2e/data/expected_metrics.json")
else
	echo "$language"
	expected_wer=$(jq --arg keyvar "$language" '.[$keyvar].accuracy' "tests/e2e/data/expected_metrics.json")
	expected_oov=$(jq --arg keyvar "$language" '.[$keyvar].oov' "tests/e2e/data/expected_metrics.json")
fi

echo "Comparing obtained and expected WER metrics..."
compare_metrics ${wer_metric} ${expected_wer}

echo "Comparing obtained and expected OOV metrics..."
compare_metrics ${oov_metric} ${expected_oov}

rm "test_${language}_results.tsv"
rm "test_${language}_oov.json"

if [ $TEST_PASSED == false ];
then
	echo "Test did not pass"
	exit 1;
fi