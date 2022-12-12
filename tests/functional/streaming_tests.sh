# Sends audio without header chunk
export RESPONSE=$(ghz --config streaming_test_configuration.toml -D streaming_test_requests.noheader.json)
echo $RESPONSE
export OK7=$(echo $RESPONSE | jq ".statusCodeDistribution.OK")
test $OK7 -eq 1 && echo "Success 9" || echo "Test 9 Failed."