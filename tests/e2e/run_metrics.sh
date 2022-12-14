#!/bin/bash

language=$1
AWS_IP=$2
gui=$3
TEST_PASSED=true
variant="none"
interval=0.05

DIALECTS_ES=("es" "mx" "co" "pe" "us")


if [[ $language = @(es-es|es-mx|es-co|es-pe|es-us) ]]; then
	variant=$(echo $language | cut -d "-" -f 2)
	language="es"
fi

function compare_metrics(){
	metric="$1"
	expected_metric="$2"

	if (( $(echo "$metric+$interval >= $expected_metric" |bc -l) )) && (( $(echo "$metric-$interval <= $expected_metric" |bc -l) ));
	then
	  echo "Obtained values and expected values match (${metric})"
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
python bin/client.py -l "${language}" --host "${AWS_IP}" -g "${gui}" -m -v DEBUG
sleep 10

accuracy_metric=$(cat "test_${language}_results.tsv" | grep "Accuracy" | cut -d " " -f 2)
oov_metric=$(jq '.score' "test_${language}_oov.json" | sed 's/[^0-9.]*//g')

expected_accuracy=$(jq --arg keyvar "$language" '.[$keyvar].accuracy' "tests/e2e/data/expected_metrics.json")
expected_oov=$(jq --arg keyvar "$language" '.[$keyvar].oov' "tests/e2e/data/expected_metrics.json")

echo "Comparing obtained and expected accuracy metrics of $language..."
compare_metrics ${accuracy_metric} ${expected_accuracy}

echo "Comparing obtained and expected OOV metrics..."
compare_metrics ${oov_metric} ${expected_oov}

if [ "${language}" == "es" ];
then
	for dialect in  "${DIALECTS_ES}";
	do
		accuracy_metric=$(jq --arg dialect "$language-$dialect" '.[$dialect]' "test_${language}_intratest/dialects_intratest.json")
		expected_accuracy=$(jq --arg lang "$language" --arg dialect "$language-$dialect" '.[$lang].dialects[$dialect].accuracy' "tests/e2e/data/expected_metrics.json")

		echo "Comparing obtained and expected accuracy metrics of $language-$dialect..."
		compare_metrics ${accuracy_metric} ${expected_accuracy}

	done

	deviation_metric=$(jq '."Accuracy typical deviation"' "test_${language}_intratest/dialects_intratest.json")
	expected_deviation=$(jq --arg lang "$language" '.[$lang].dialects["typical_deviation"]' "tests/e2e/data/expected_metrics.json")
	
	echo "Comparing obtained and expected accuracy deviation metrics betweens $language dialects..."
	compare_metrics ${deviation_metric} ${expected_deviation}
fi

while read domain;
do
	accuracy_metric=$(jq --arg dom "$domain" '.[$dom]' "test_${language}_intratest/domains_intratest.json")
	expected_accuracy=$(jq --arg lang "$language" --arg dom "$domain" '.[$lang].domains[$dom].accuracy' "tests/e2e/data/expected_metrics.json")

	echo "Comparing obtained and expected accuracy metrics of $domain..."
	compare_metrics ${accuracy_metric} ${expected_accuracy}

 done < "tests/e2e/data/domains_$language.txt"

deviation_metric=$(jq '."Accuracy typical deviation"' "test_${language}_intratest/domains_intratest.json")
expected_deviation=$(jq --arg lang "$language" '.[$lang].domains["typical_deviation"]' "tests/e2e/data/expected_metrics.json")

echo "Comparing obtained and expected accuracy deviation metrics between domains..."
compare_metrics ${deviation_metric} ${expected_deviation}

rm "test_${language}_results.tsv"
rm "test_${language}_oov.json"
rm -rf "test_${language}_intratest"
rm -rf "wer"

if [ $TEST_PASSED == false ];
then
	echo "Test did not pass"
	exit 1;
fi
