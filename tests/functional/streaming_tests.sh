

# Test complete stream in 8 parts
export OK1=$(ghz --config streaming_test_configuration.toml -D streaming_test_requests.8.json | jq ".statusCodeDistribution.OK")
test $OK1 -eq 1 && echo "Success 1" || echo "Test 1 Failed."
    
# Test only half of the parts
export OK2=$(ghz --config streaming_test_configuration.toml -D streaming_test_requests.4.json | jq ".statusCodeDistribution.OK")
test $OK2 -eq 1 && echo "Success 2" || echo "Test 2 Failed."

# Sends several "config" parameters changing the sample rate (right)
export OK3=$(ghz --config streaming_test_configuration.toml -D streaming_test_requests.reconfig.json | jq ".statusCodeDistribution.OK")
test $OK3 -eq 1 && echo "Success 3" || echo "Test 3 Failed."

# Sends several "config" parameters changing the sample rate (wrong)
export OK4=$(ghz --config streaming_test_configuration.toml -D streaming_test_requests.error1.json | jq ".statusCodeDistribution.OK")
test $OK4 -eq 1 && echo "Success 4" || echo "Test 4 Failed."

# Sends audio with no header
export UNK1=$(ghz --config streaming_test_configuration.toml -D streaming_test_requests.error2.json | jq ".statusCodeDistribution.Unknown")
test $UNK1 -eq 1 && echo "Success 5" || echo "Test 5 Failed."

# Sends header with no audio
export UNK2=$(ghz --config streaming_test_configuration.toml -D streaming_test_requests.error3.json | jq ".statusCodeDistribution.Unknown")
test $UNK2 -eq 1 && echo "Success 6" || echo "Test 6 Failed."

# Fast sending
export OK5=$(ghz --config streaming_test_configuration.toml -D streaming_test_requests.8.json --stream-interval=1ms --concurrency 10 --total 10 | jq ".statusCodeDistribution.OK")
test $OK5 -eq 10 && echo "Success 7" || echo "Test 7 Failed."

# Slow sending
export OK6=$(ghz --config streaming_test_configuration.toml -D streaming_test_requests.8.json --stream-interval=10s | jq ".statusCodeDistribution.OK")
test $OK6 -eq 1 && echo "Success 8" || echo "Test 8 Failed."

# Sends audio without header chunk
export OK7=$(ghz --config streaming_test_configuration.toml -D streaming_test_requests.noheader.json | jq ".statusCodeDistribution.OK")
test $OK7 -eq 1 && echo "Success 9" || echo "Test 9 Failed."

# Send slower than timeout
export ERR1=$(ghz --config streaming_test_configuration.toml -D streaming_test_requests.8.json -t 2s --stream-interval=1s | jq ".statusCodeDistribution.DeadlineExceeded")
test $ERR1 -eq 1 && echo "Success 10" || echo "Test 10 Failed."



#test $OK2 -eq 1 && test $OK3 -eq 1 && test $OK4 -eq 1 && test $OK5 -eq 10 && test $OK6 -eq 1 && test $OK7 -eq 1 && test $ERR1 -eq 1 && test $UNK1 -eq 1 && test $UNK2 -eq 1
test $OK2 -eq 1 && test $OK3 -eq 1 && test $OK4 -eq 1 && test $OK5 -eq 10 && test $OK6 -eq 1 && test $ERR1 -eq 1 && test $UNK1 -eq 1 && test $UNK2 -eq 1