# Python integration with the Verbio Speech Center cloud.

This repository contains a python based example of how to use the Verbio Technologies Speech Center cloud.

## Requirements

### Starting requirements 
Currently, we support Python 3.6+.

Before to start you will need:

1. Speech Center proto file (provided in this repository)
2. Platform access token (provided to you by Verbio Technologies)
3. Speech Center endpoint (https://www.speech-center.verbio.com:2424)


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
On a directory containing the .proto file provided by Verbio, run the following shell commands:
```commandline
# In case you want to use our Speech Center Recognizer
generate_grpc_code.sh verbio-speech-center-recognizer.proto ./ ./

# In case you want to use our Speech Center Synthesizer
generate_grpc_code.sh verbio-speech-center-synthesizer.proto ./ ./
```

This will generate a set of python files with grpc calls:
```commandline
# In case you generated the Speech Center Recognizer files
verbio_speech_center_recognizer_pb2.py
verbio_speech_center_recognizer_pb2_grpc.py

# In case you generated the Speech Center Synthesizer files
verbio_speech_center_synthesizer_pb2.py
verbio_speech_center_synthesizer_pb2_grpc.py
```

### Run a client

The CLI clients will use the generated code to connect to the speech center cloud to process your speech file or synthesize your input.  
  

#### Recognizer

Our Recognizer will allow you to easily convert an audio resource into its associated text. In order to run the CLI Speech Center Recognizer client, check out the following commands:

```commandline
.>$ cd src/cli-client/
./src/cli-client>$ ./recognizer.py --help
usage: recognizer.py [-h] --audio-file AUDIO_FILE (--grammar GRAMMAR | --topic {GENERIC,TELCO,BANKING}) [--language {en-US,pt-BR,es-ES}] --token TOKEN [--host HOST]

Perform speech recognition on an audio file

optional arguments:
  -h, --help            show this help message and exit
  --audio-file AUDIOFILE, -a AUDIOFILE
                        Path to a .wav audio in 8kHz and PCM16 encoding
  --grammar GRAMMAR, -g GRAMMAR
                        Path to a file containing an ABNF grammar
  --topic {GENERIC,TELCO,BANKING}, -T {GENERIC,TELCO,BANKING}
                        A valid topic
  --language {en-US,pt-BR,es-ES}, -l {en-US,pt-BR,es-ES}
                        A Language ID (default: en-US)
  --token TOKEN, -t TOKEN
                        A string with the authentication token
  --host HOST, -H HOST  The URL of the host trying to reach (default: speechcenter.verbio.com:2424)
```

**Example**

```commandline
python3 ./recognizer.py --audio-file file.wav --topic GENERIC --token you.speech-center.token
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
.>$ cd src/cli-client/
./src/cli-client>$ ./synthesizer.py --help
usage: synthesizer.py [-h] --text TEXT --voice {Tommy,Annie,Aurora,Luma,David} [--sample-rate {8000}] [--encoding {PCM}] [--format {wav,raw}] [--language {en-US,pt-BR,es-ES}] --token TOKEN [--host HOST] --audio-file AUDIO_FILE

Perform speech recognition on an audio file

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
  --language {en-US,pt-BR,es-ES}, -l {en-US,pt-BR,es-ES}
                        A Language ID (default: en-US)
  --token TOKEN, -t TOKEN
                        A string with the authentication token
  --host HOST, -H HOST  The URL of the host trying to reach (default: speechcenter.verbio.com:2424)
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
* David [es-ES, ca-CA]
