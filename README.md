# asr4

ASR based on Transformer DNNs, with multilingual and unsupervised information.

## How to Install

The installation is divided into two parts: installing the server or the client. The client has fewer dependencies than the server.  

**If you only want to quickly perform a test inference, we recommend that you just install the client and run it against the server located in tanzania.**

### Client installation (HTTP)

The only requirements for the HTTP client are `cURL`, `jq` and `base64`. In case your system does not include `cURL` or `jq` by default run the following command:

```sh
# Ubuntu/Debian
apt-get update
apt-get install -y curl jq

# RHEL/CentOS/Fedora
yum install -y curl jq
```

:warning: `jq` is highly recommended but not actually required. You could for instance store the base64-encoded audio contents into a variable and use it on the `cURL` command, but you would have to face UNIX maximum length permitted for a command.

### Client installation (gRPC)

Python version 3.7+ is required to run the client. It is recommended to update the pip package:

```sh
pip install --upgrade pip
```

To install the client's requirements, run the following command from the root of the `asr4` repository:

```sh
pip install .[client]
```

Check if asr4 is correctly installed by printing its version.

```sh
python -c 'import asr4; print(asr4.__version__)'
```

### Server installation

Python version 3.7+ is required to run the server. 
To install the server's requirements, run the following command from the root of the `asr4` repository:

```sh
pip install torch==1.12.1+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install .[server]
```

### Uninstall asr4

First remove the folder "asr4.egg-info" from the asr4 repository root and then run:
```sh
pip uninstall asr4
```

## How to Run

Again, separated instructions are provided to either run the client or the server. If you wish to perform a quick test inference just take a look at the client sub-section.

### Client (HTTP)


Once its dependencies are installed, the client can connect to a running `asr4` server to obtain transcriptions. This simple command will return the transcription through the standard output channel:

```sh
base64 <WAV_FILE_PATH> -w 0 | jq -Rs '{audio: .}' | curl -d @- \
  --url "http://tanzania:50051/v1/recognize?config.parameters.language=en-US&config.parameters.sampleRateHz=16000&config.resource.topic=GENERIC" \
  -H 'Accept-Language: en-US'
```

Notice the URL encodes the following parameters: `language` (en-US), `sampleRateHz` (16000) and `topic` (GENERIC). **Additionally, `Accept-Language` header is used to indicate the content's language to the traffic router**.

Alternatively, you can create a JSON file containing all the data:

```sh
jq -n \
  --slurpfile audio <(base64 <WAV_FILE_PATH> -w 0 | jq -Rs .) \
  '{config: {parameters: {language: "en-US", sampleRateHz: 16000}, resource: {topic: "GENERIC"}}, audio: $audio[0]}' \
  | curl http://tanzania:50051/v1/recognize -H 'Accept-Language: en-US' -d @-
```

Find supported configuration options in the table below:

|Option|Supported Values|
|-|-|
|Language|en-US, es, pt-BR|
|Sample Rate (Hz)| 16000, 8000|
|Recognition Topic| GENERIC|

### Client (gRPC)

Once its dependencies are installed, the client can connect to a running `asr4` server to obtain transcriptions. This simple command will return the transcription through the standard output channel:

```sh
# Send a recognition request against the asr4 server located at tanzania and obtain the transcription of a WAV audio
python bin/client.py --host tanzania:50051 -l en-US -a <WAV_FILE_PATH>.wav -o <OUTPUT_PATH>

# Or the transcription of several audios in a .gui file
python bin/client.py --host tanzania:50051 -l en-US -g <GUI_FILE_PATH>.gui -o <OUTPUT_PATH>

# Get metrics after the trascription process has finished
python bin/client.py --host tanzania:50051 -l en-US -g <GUI_FILE_PATH>.gui -o <OUTPUT_PATH> -m

# Try again with a Spanish WAV audio
python bin/client.py --host tanzania:50051 -l es -a <WAV_FILE_PATH>.wav -o <OUTPUT_PATH>

```

Note that it needs to define `PYTHONPATH` to the root of the repo to work.

Additionally, include the `--help` argument to display all available options:

```
» python bin/client.py --help
usage: client.py [-h] [-o OUTPUT] (-a AUDIO | -g GUI) [-l {en-us,es,pt-br}] [--host HOST] [-j JOBS] [-m] [-v VERBOSE]

A Speech Recognition client.

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT, --output-dir OUTPUT
                        Output path for the results.
  -a AUDIO, --audio-path AUDIO
                        Path to the audio file.
  -g GUI, --gui-path GUI
                        Path to the gui file with audio paths.
  -l {en-us,es,pt-br}, --language {en-us,es,pt-br}
                        Language of the recognizer service.
  --host HOST           Hostname address of the ASR4 server.
  -j JOBS, --jobs JOBS  Number of parallel workers; if not specified, defaults to CPU count.
  -m, --metrics         Calculate metrics using the audio transcription references.
  -v VERBOSE, --verbose VERBOSE
                        Log levels. Options: CRITICAL, ERROR, WARNING, INFO and DEBUG.

```

Find supported configuration options in the table below:

|Option|Supported Values|
|-|-|
|Language|*en-US, *es, *pt-BR|
|Sample Rate (Hz)| 8000, 16000|
|Recognition Topic| GENERIC|

\***Case Insensitive**

### Server

Once its dependencies are installed, the `asr4` can be executed to accept connections from the clients. This simple command will return the transcription through the standard output channel:

```sh
# Run a single-threaded asr4 server that serves an ONNX model.
python bin/server.py -m <ONNX_FILE_PATH>.onnx -j1

# Run a single-threaded asr4 server that servers an ONNX model and formats the transcription.
python bin/server.py -m <ONNX_FILE_PATH>.onnx -f <FORMATTER_FILE_PATH>.fm -j1
```

Note that right now both transcription and formatting models can only be obtained from barrayar.

Additionally, include the `--help` argument to display all available options:

```
» python bin/server.py --help
usage: server.py [-h] -m MODEL [-d VOCABULARY] [-l {en-us,es,pt-br}] [-f FORMATTER]
                 [--host BINDADDRESS] [-j JOBS] [-v VERBOSE]

Python ASR4 Server

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model-path MODEL
                        Path to the model file.
  -d VOCABULARY, --dictionary-path VOCABULARY
                        Path to the model's dictionary file, containing all the possible outputs from
                        the model.
  -l {en-us,es,pt-br}, --language {en-us,es,pt-br}
                        Language of the recognizer service.
  -f FORMATTER, --formatter-model-path FORMATTER
                        Path to the formatter model file.
  --host BINDADDRESS    Hostname address to bind the server to.
  -j JOBS, --jobs JOBS  Number of parallel workers; if not specified, defaults to CPU count.
  -v VERBOSE, --verbose VERBOSE
                        Log levels. Options: CRITICAL, ERROR, WARNING, INFO and DEBUG. By default reads env variable LOG_LEVEL.
```


## Formatting & Linting

As a general note, you should always apply proper formatting and linting before pushing a commit. To do so, please run the following:

```sh
# Rust
cargo fmt --all # Formatting
cargo clippy --all --all-targets -- -D warnings # Linting

# Python
black . # Formatting & Linting
```

## How to generate a new release

In order to manage releases we will use `cargo-release`: https://github.com/sunng87/cargo-release

If you want to install it, run the following command:

```
$ cargo install cargo-release
```

In order to generate a new release, you can invoke the following command (the flag `--workspace` is required in order to update all the subcrates):

```
# Increment major
$ cargo release major --workspace

# Increment minor
$ cargo release minor --workspace

# Increment patch
$ cargo release patch --workspace

# Set specific version
$ cargo release 1.2.3 --workspace 
```

It will:

1. Set the new version to all the `Cargo.toml` of the main crate and subcrates. It also updates the `Cargo.lock`.
2. Set the new version to the `CMakeLists.txt` file.
3. Update the `VERSION` file.
4. Create a "Bump version" commit.

⚠ NOTE: IT DOES NOT ACTUALLY PERFORM ANY MODIFICATION UNLESS THE FLAG `--execute` IS PROVIDED
⚠ NOTE: IT DOES NOT CREATE THE TAG NOR PUSH ANYTHING

So, a typical workflow would be the following:

```
$ cargo release patch --workspace --execute
$ git push 
$ git tag VERSION_NUMBER
$ git push --tags 
```
The user must be able to perform both the `git push` to upload the commit and the `git push --tags` to upload the newly created tag.

Once the tag has been pushed, you should edit the tag's Release notes via the GitLab UI and add the changelog for the version.

Then, the CI system will start doing its job and it will generate a `.deb` package, and via webhook the package will be available at the Verbio staging repositories.
