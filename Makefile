CI ?= false
PWD := $(dir $(abspath $(firstword $(MAKEFILE_LIST))))

## VENV
VENV_DIR = .venv/
VENV_BIN ?= $(PWD)$(VENV_DIR)bin/

## Installed executables from pyproject
PYTHON ?= $(VENV_BIN)python

TOKEN_FILE = token.file
VERBIO_CREDENTIALS ?= ${HOME}/.verbio/credentials

SOURCE_DIR = $(PWD)cli-client/
STT = $(PYTHON) $(SOURCE_DIR)recognizer_stream.py
TTS = $(PYTHON) $(SOURCE_DIR)synthesizer_stream.py


VERBIO_URL ?= pre.eu.verbiospeechcenter.com
CLIENT_ID ?= $(shell cat $(VERBIO_CREDENTIALS) | jq .client_id -r)
CLIENT_SECRET ?= $(shell cat $(VERBIO_CREDENTIALS) | jq .client_secret -r)
TOPIC ?= GENERIC
ASR_VERSION ?= V2
TTS_AUDIO_FILE ?= tts_$(shell date +%Y-%m-%d_%H-%M-%S).wav


# Add .env environment variables unless is running inside of CI pipeline
-include $(PWD)/.env
export

#Include other makefiles
-include *.mk

## Target convention naming:
## <Action>[-<Identifier>] :: Examples:
### `install` -> Just the action because is a generic task may implies other tasks.
### `install-poetry` -> The action first, the name after.
### `build-docker` -> Action and identifier.
### `rm-build-docker` -> Action taken for a action result.
## Why?
# Because helps to find the correct targets using the Shell AutoCompletion.

.PHONY: install proto help help-stt help-tts stt tts guard-% check-cmd-%

check-cmd-%:
	@which $* > /dev/null 2>&1 || (echo "ERROR: '$*' is not installed." && exit 1)

install: | $(VENV_DIR) proto
	$(PIP) install -r requirements-dev.txt

proto: check-cmd-buf
	buf generate

$(VENV_DIR): check-cmd-python3
	python3 -mvenv $(VENV_DIR)

$(TOKEN_FILE):
	touch $(TOKEN_FILE)

help-stt:
	@echo "Target: stt — Speech-to-text recognition"
	@echo ""
	@echo "  Required:"
	@echo "    LANGUAGE=<lang>              Language ID (en-US, es, es-ES, pt-BR, ca-ES, ...)"
	@echo "    AUDIO_FILE=<path>            Path to .wav audio file (8kHz, PCM16)"
	@echo ""
	@echo "  Optional:"
	@echo "    TOPIC=<topic>                GENERIC|TELCO|BANKING|INSURANCE (default: $(TOPIC))"
	@echo "    ASR_VERSION=<ver>            V1|V2 (default: $(ASR_VERSION))"
	@echo "    INLINE_GRAMMAR=<str>         Grammar inline as a string"
	@echo "    GRAMMAR_URI=<uri>            Builtin grammar URI"
	@echo "    COMPILED_GRAMMAR=<path>      Compiled grammar file (.tar.xz)"
	@echo "    DIARIZATION=1                Enable diarization"
	@echo "    FORMATTING=1                 Enable formatting"
	@echo "    HIDE_PARTIAL_RESULTS=1       Hide partial transcription results"
	@echo "    INACTIVITY_TIMEOUT=<sec>     Stream inactivity timeout"
	@echo "    LABEL=<label>                Label for the request"
	@echo "    WORD_BOOSTING='w1 w2 ...'    Words to boost during recognition"
	@echo "    CONVERT_AUDIO=1              Convert A-LAW audio to PCM"
	@echo "    NOT_SECURE=1                 Disable secure channel"
	@echo ""
	@echo "  Example:"
	@echo "    make stt LANGUAGE=es AUDIO_FILE=./audio.wav"

help-tts:
	@echo "Target: tts — Text-to-speech synthesis"
	@echo ""
	@echo "  Required:"
	@echo "    VOICE=<voice>                Voice for synthesis (e.g. david_es_es, marvin_en_us)"
	@echo "    TEXT=<text>                  Text to synthesize"
	@echo "    or TEXT_FILE=<path>          File with newline-delimited text to synthesize"
	@echo ""
	@echo "  Optional:"
	@echo "    TTS_AUDIO_FILE=<path>        Output audio file (default: tts_<timestamp>.wav)"
	@echo "    SAMPLE_RATE=<rate>           8000|16000"
	@echo "    FORMAT=<fmt>                 wav|raw"
	@echo "    INACTIVITY_TIMEOUT=<sec>     Stream inactivity timeout"
	@echo "    NOT_SECURE=1                 Disable secure channel"
	@echo ""
	@echo "  Example:"
	@echo "    make tts VOICE=david_es_es TEXT='Hola mundo'"

help:
	@echo "Usage: make <target> [VARIABLE=value ...]"
	@echo ""
	@echo "Available targets:"
	@echo "  install      Install dependencies"
	@echo "  proto        Generate protobuf files"
	@echo "  stt          Speech-to-text recognition"
	@echo "  tts          Text-to-speech synthesis"
	@echo "  help-stt     Show STT usage and variables"
	@echo "  help-tts     Show TTS usage and variables"
	@echo ""
	@echo "Common variables:"
	@echo "  VERBIO_URL=<url>    Service URL (default: $(VERBIO_URL))"
	@echo ""
	@$(MAKE) --no-print-directory help-stt
	@echo ""
	@$(MAKE) --no-print-directory help-tts

guard-%:
	@[ -n "${$*}" ] || (echo "ERROR: $* is required. Usage: make stt LANGUAGE=es AUDIO_FILE=./audio.wav" && exit 1)

stt: $(TOKEN_FILE) guard-LANGUAGE guard-AUDIO_FILE
	@$(STT) --token $(TOKEN_FILE) --client-id $(CLIENT_ID) --client-secret $(CLIENT_SECRET) \
		--host $(VERBIO_URL) \
		--topic $(TOPIC) \
		--asr-version $(ASR_VERSION) \
		--language $(LANGUAGE) \
		--audio-file $(AUDIO_FILE) \
		$(if $(INLINE_GRAMMAR),--inline-grammar $(INLINE_GRAMMAR)) \
		$(if $(GRAMMAR_URI),--grammar-uri $(GRAMMAR_URI)) \
		$(if $(COMPILED_GRAMMAR),--compiled-grammar $(COMPILED_GRAMMAR)) \
		$(if $(CONVERT_AUDIO),--convert-audio) \
		$(if $(NOT_SECURE),--not-secure) \
		$(if $(DIARIZATION),--diarization) \
		$(if $(FORMATTING),--formatting) \
		$(if $(HIDE_PARTIAL_RESULTS),--hide-partial-results) \
		$(if $(INACTIVITY_TIMEOUT),--inactivity-timeout $(INACTIVITY_TIMEOUT)) \
		$(if $(LABEL),--label $(LABEL)) \
		$(if $(WORD_BOOSTING),--word-boosting $(WORD_BOOSTING))

tts: $(TOKEN_FILE) guard-VOICE
	@$(TTS) --token $(TOKEN_FILE) --client-id $(CLIENT_ID) --client-secret $(CLIENT_SECRET) \
		--host $(VERBIO_URL) \
		--voice $(VOICE) \
		--audio-file $(TTS_AUDIO_FILE) \
		$(if $(TEXT),--text "$(TEXT)") \
		$(if $(TEXT_FILE),--text-file $(TEXT_FILE)) \
		$(if $(SAMPLE_RATE),--sample-rate $(SAMPLE_RATE)) \
		$(if $(FORMAT),--format $(FORMAT)) \
		$(if $(NOT_SECURE),--not-secure) \
		$(if $(INACTIVITY_TIMEOUT),--inactivity-timeout $(INACTIVITY_TIMEOUT))

