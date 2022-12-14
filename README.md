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
grpcio==1.48.0
grpcio-tools==1.41.1
```
they are already write in the requirements.txt in this repository.

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
```commandline
.>$ cd scripts/
./scripts>$ ./generate_grpc_code.sh

```
It will generate all needed grpc files on the project root directory `proto/generated` like:

```commandline
verbio_speech_center_recognizer_pb2.py
verbio_speech_center_recognizer_pb2_grpc.py
...
```

### Run a client

The CLI clients will use the generated code to connect to the speech center cloud to process your speech file or synthesize your input.  
  

#### Recognizer batch and stream

Our Recognizer will allow you to easily convert an audio resource into its associated text. In order to run the CLI Speech Center Recognizer client, check out the following commands:

**Example**

```commandline
python3 ./recognizer.py --audio-file file.wav --topic GENERIC --token you.speech-center.token.file

or for a streaming version

python3 ./recognizer_stream.py --audio-file file.wav --topic GENERIC --token you.speech-center.token.file

```

This code will generate the following terminal output on success:
```commandline
INFO:root:Running Recognizer inference example...
INFO:root:Sending message RegonitionInit
INFO:root:Sending message Audio
INFO:root:Inference response: 'this is a test sentence' [status=StatusCode.OK]
```
with response.text as your speech recognition inference response and response.status as the result of the process.  
  

#### Synthesizer

Our Synthesizer will convert your text into speech. In order to run the CLI Speech Center Synthesizer client, check out the following commands:

```commandline
.>$ cd cli-client
cli-client>$ ./synthesizer.py --help
usage: synthesizer.py [-h] --text TEXT --voice {Tommy,Annie,Aurora,Luma,David} [--sample-rate {8000}] [--encoding {PCM}] [--format {wav,raw}] [--language {en-US,pt-BR,es-ES}] --token TOKEN [--host HOST] --audio-file AUDIO_FILE

Perform speech synthesis on a sample text

optional arguments:
  -h, --help            show this help message and exit
  --text TEXT, -T TEXT  Text to synthesize to audio
  --voice {Tommy,Annie,Aurora,Luma,David}, -v {Tommy,Annie,Aurora,Luma,David}
                        Voice to use for the synthesis
  --sample-rate {8000}, -s {8000}
                        Output audio sample rate in Hz (default: 8000)
  --encoding {PCM}, -e {PCM}
                        Output audio encoding algorithm (default: PCM [Signed 16-bit little endian PCM])
  --format {wav,raw}, -f {wav,raw}
                        Output audio header. (default: wav)
  --language {en-US,pt-BR,es-ES,ca-ES}, -l {en-US,pt-BR,es-ES,ca-ES}
                        A Language ID (default: en-US)
  --token TOKEN, -t TOKEN
                        A string with the authentication token
  --host HOST, -H HOST  The URL of the host trying to reach (default: tts.api.speechcenter.verbio.com)
  --audio-file AUDIO_FILE, -a AUDIO_FILE
                        Path to store the resulting audio
```

**Example**

```commandline
python3 ./synthesizer.py --text "this is a test sentence" --voice Tommy --token you.speech-center.token --audio-file file.wav
```

This code will generate the following terminal output on success:
```commandline
INFO:root:Running Synthesizer inference example...
INFO:root:Sending message SynthesisRequest
INFO:root:Inference response [status=StatusCode.OK]
INFO:root:Stored resulting audio at file.wav
```
with response.audio as your speech synthesis infenrece response stored in the 'file.wav' `audio-file` and response.status as the result of the process.

**Notice**

As a notice of the Speech Center Synthesizer, not all Voice / Language combinations are available. Currently, we support the following ones:

* Tommy [en-US]
* Annie [en-US]
* Aurora [es-ES]
* Luma [pt-BR]
* David [es-ES, ca-ES]
