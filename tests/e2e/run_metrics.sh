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

function compare_metrics(){
	metric="$1"
	expected_metric="$2"
	comparison=$3

	if (( $(echo "$metric+$interval >= $expected_metric" |bc -l) )) && (( $(echo "$metric-$interval <= $expected_metric" |bc -l) ));
	then
		echo "$comparison: Obtained values and expected values match (${metric})"
	elif (( $(echo "$metric > $expected_metric" |bc -l) ));
	then
		echo "$comparison: Obtained value (${metric}) is higher than expected value (${expected_metric})"
	else 
		echo "$comparison: Obtained value (${metric}) is lower than expected value (${expected_metric})"
		TEST_PASSED=false
	fi
}

pip install .[client]
python bin/client.py -l "${language}" --host "${AWS_IP}" -g "${gui}" -m 
sleep 10

if [ -f "test_${language}_results.tsv" ]; then

	accuracy_metric=$(cat "test_${language}_results.tsv" | grep "Accuracy" | cut -d " " -f 2 )
	expected_accuracy=$(jq --arg testtype "$test" --arg lang "$language" '.[$lang][$testtype].accuracy' "tests/e2e/data/expected_metrics.json")
	compare_metrics ${accuracy_metric} ${expected_accuracy} "${language} accuracy metrics"

	oov_metric=$(jq '.score' "test_${language}_oov.json" | sed 's/[^0-9.]*//g')
	expected_oov=$(jq --arg testtype "$test" --arg lang "$language" '.[$lang][$testtype].oov' "tests/e2e/data/expected_metrics.json")
	compare_metrics ${oov_metric} ${expected_oov} "${language} OOV metrics"

	if [ "${language}" == "es" ];
	then
		dialects=( $(cat  "tests/e2e/data/expected_metrics.json" | jq --arg testtype "$test" --arg lang "$language" '.[$lang][$testtype].dialects | keys[]' | sed 's/"//g') )
		for dialect in  ${dialects[@]};
		do
			accuracy_metric=$(jq --arg dialect "$dialect" '.[$dialect]' "test_${language}_intratest/dialects_intratest.json")
			expected_accuracy=$(jq --arg testtype "$test" --arg lang "$language" --arg dialect "$dialect" '.[$lang][$testtype].dialects[$dialect].accuracy' "tests/e2e/data/expected_metrics.json")

			compare_metrics ${accuracy_metric} ${expected_accuracy} "$dialect accuracy metrics"

		done

		deviation_metric=$(jq '."Accuracy typical deviation"' "test_${language}_intratest/dialects_intratest.json")
		expected_deviation=$(jq --arg testtype "$test" --arg lang "$language" '.[$lang][$testtype].dialects_typical_deviation' "tests/e2e/data/expected_metrics.json")
		
		compare_metrics ${deviation_metric} ${expected_deviation} "$language dialect accuracy deviation metrics"
	fi

	domains=( $(cat  "tests/e2e/data/expected_metrics.json" | jq --arg testtype "$test" --arg lang "$language" '.[$lang][$testtype].domains | keys[]' | sed 's/"//g') )
	for domain in ${domains[@]};
	do
		accuracy_metric=$(jq --arg dom $domain '.[$dom]' "test_${language}_intratest/domains_intratest.json")
		expected_accuracy=$(jq --arg testtype "$test" --arg lang "$language" --arg dom $domain '.[$lang][$testtype].domains[$dom].accuracy' "tests/e2e/data/expected_metrics.json")

		compare_metrics ${accuracy_metric} ${expected_accuracy} "$domain accuracy metrics"

	done

	deviation_metric=$(jq '."Accuracy typical deviation"' "test_${language}_intratest/domains_intratest.json")
	expected_deviation=$(jq --arg testtype "$test" --arg lang "$language" '.[$lang][$testtype].domains_typical_deviation' "tests/e2e/data/expected_metrics.json")

	compare_metrics ${deviation_metric} ${expected_deviation} "$language domain accuracy deviation metrics"

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
