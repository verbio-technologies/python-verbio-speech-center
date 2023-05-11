# Python integration with the Verbio Speech Center cloud.

This repository contains a python example of how to use the Verbio Technologies Speech Center cloud.

## Requirements

### Starting requirements 
Currently, we support Python 3.6+.

Before to start you will need:

1. Speech Center proto files provided under proto directory.
2. Platform access token (provided to you by Verbio Technologies)
3. Speech Center endpoint



### Python packages:
```requirements.txt
protobuf==3.19.1
grpcio==1.51.1
grpcio-tools==1.41.1
requests==2.26.0
pyjwt==2.6.0
```
they are already written in the requirements.txt in this repository.

The grpc and protobuf packages are necessary to automatically generate from the .proto specification all the necessary code that the main python script will use to connect with the gRCP server in the cloud.

##  Step by step
The steps needed are very similar to the ones described in the grpc official guide.

### Install dependencies
You can use the standard pip call to install all the necessary dependencies:
```commandline
pip install -r requirements.txt
```

### Generate grpc code with python
In scritps repository there is a `generate_grpc_code.sh` script that will generate the gRPC and Protobuf code for you.
```console
.>$ cd scripts/
./scripts>$ ./generate_grpc_code.sh
```

It will generate all needed grpc files on the project root directory `proto/generated` like:

```console
verbio_speech_center_recognizer_pb2.py
verbio_speech_center_recognizer_pb2_grpc.py
...
```

### Run a client

The CLI clients will use the generated code to connect to the speech center cloud to process your speech file.  
  

#### Recognizer stream

Our Recognizer will allow you to easily convert an audio resource into its associated text. In order to run the CLI Speech Center Recognizer client, check out the following command:

**Example**

```console
python3 speechcenter/cli-client.py --audio-file file.wav --topic GENERIC --language en-US --host us.speechcenter.verbio.com --token token.file --asr-version V1 --label project1
```

This code will generate the following terminal output on success:
```console
[2023-04-04 12:28:29,078][INFO]:Running speechcenter streaming channel...
[2023-04-04 12:28:29,080][INFO]:Connecting to us.speechcenter.verbio.com
[2023-04-04 12:28:29,082][INFO]:Running executor...
[2023-04-04 12:28:29,083][INFO]:Sending streaming message config
[2023-04-04 12:28:29,083][INFO]:Running response watcher
[2023-04-04 12:28:29,083][INFO]:Waiting for server to respond...
[2023-04-04 12:28:29,084][INFO]:Sending streaming message audio
[2023-04-04 12:28:29,084][INFO]:All audio messages sent
[2023-04-04 12:31:27,109][INFO]:New incoming response: '{  "result": {    "alternatives": [      {     ...'
	"transcript": "Hi. My name is John Doe.",
	"confidence": "0.899752",
	"duration": "4.460000"
[2023-04-04 12:31:27,110][INFO]:New incoming response: '{  "result": {    "alternatives": [      {     ...'
	"transcript": "I'd like to check my account balance, please.",
	"confidence": "0.994662",
	"duration": "7.000000"
[2023-04-04 12:31:32,111][INFO]:Stream inactivity detected, closing stream...
[2023-04-04 12:31:32,112][INFO]:Recognition finished
```

You can also run:
```console
python3 speechcenter/cli-client.py --help
```
  
to list all the available options.

## Automatically Refresh Service Token
This repository optionally implements an automatic token update. To do so, you must specify your credentials (find them in the Client Credentials section of the [user dashboard](https://dashboard.speechcenter.verbio.com)).

You must also specify a token file, where the token will be stored and updated in case it is invalid or expired.

**Example**
```console
python3 speechcenter/cli-client.py --client-id="your-client-id" --client-secret="your-client-secret"
 --audio-file file.wav --topic GENERIC --language en-US --host us.speechcenter.verbio.com --token token.file --asr-version V1 --label project1
```

**WARNING**

Please note that due to a limitation of the command line argument parser, the `client_id` and `client_secret` arguments MUST be written in the following format since they might contain hyphens.

```
--client-id="your-client-id"
           ^
```
> Note the usage of the `=` sign


Not defining the arguments like this will yield an error.