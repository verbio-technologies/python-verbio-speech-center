# Python integration with the Verbio Speech Center cloud.

This repository contains a python based example of how to use the Verbio Technologies Speech Center cloud.

## Requirements

### Starting requirements 
Currently, we support Python 3.6+.

Before to start you will need:

1. Speech center proto file (provided in this repository)
2. Platform access token (provided to you by Verbio Technologies)
3. Speech center endpoint (https://www.speech-center.verbio.com:2424)


### Python packages:
```requirements.txt
protobuf==3.19.1
grpcio==1.41.1
grpcio-tools==1.41.1
```
they are already write in the requirements.txt in this repository.

The grpc and protobuf packages are necessary to automatically generate from the .proto specification all the necessary code that the main python script will use to connect with the gRCP server in the cloud.

##  Step by step
The steps needed are very similar to the ones described in the grpc official guide.

### Install dependencies
You can the standard pip call to install all the necessary dependencies:
```commandline
pip install -r requirements.txt
```

### Generate grpc code with python
In this repository there is a `generate_grpc_code.sh` script that will generate the gRPC and Protobuf code for you. You can find it at `src/cli-client` folder.
```commandline
.>$ cd src/cli-client/
./src/cli-client>$ /generate_grpc_code.sh 

use: ./generate_grpc_code.sh <protobuf_definition_file> <python_output_path> <grpc_output_path>
```
On a directory containing the .proto file provided by Verbio, run the following shell command:
`generate_grpc_code.sh verbio-speech-center.proto ./ ./`

This will generate a set of python files with grpc calls:
```commandline
verbio_speech_center_pb2.py
verbio_speech_center_pb2_grpc.py
```

### Run the client

The cli_client will use the generated code to connect to the speech center cloud to process your speech file.
```commandline
.>$ cd src/cli-client/
./src/cli-client>$ ./cli_client.py --help
usage: cli_client.py [-h] --audiofile AUDIOFILE (--grammar GRAMMAR | --topic {GENERIC,TELCO,BANKING}) --token TOKEN [--host HOST]

Perform speech recognition on an audio file

optional arguments:
  -h, --help            show this help message and exit
  --audiofile AUDIOFILE, -a AUDIOFILE
                        Path to a .wav audio in 8kHz and PCM16 encoding
  --grammar GRAMMAR, -g GRAMMAR
                        Path to a file containing an ABNF grammar
  --topic {GENERIC,TELCO,BANKING}, -T {GENERIC,TELCO,BANKING}
                        A valid topic
  --token TOKEN, -t TOKEN
                        A string with the authentication token
  --host HOST, -H HOST  The URL of the host trying to reach
```

#### Example
```commandline
python3 ./cli_client.py --audiofile file.wav --topic GENERIC --token you.speech-center.token --host speechcenter.verbio.com:2424
```

This code will generate the following terminal output on success:
```commandline
INFO:root:Sending message...
INFO:root:Sending message...
INFO:root:Speech recognition response: 'this is a test sentence'
INFO:root:Speech recognition call status:
INFO:root:<_MultiThreadedRendezvous of RPC that terminated with:
	status = StatusCode.OK
	details = "">
```
with response.text as your speech recognition result and response.status as the result of the process.
