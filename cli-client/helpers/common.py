import math
import argparse
import logging
from helpers.speechcenterauth import SpeechCenterCredentials


class SynthesizerOptions:
    def __init__(self):
        self.token_file = None
        self.host = ""
        self.audio_file = None
        self.secure_channel = True
        self.audio_format = ''
        self.sample_rate: int = 0
        self.voice: str = None
        self.text: str = ''
        self.text_file: str = ''
        self.inactivity_timeout = False
        self.client_id = None
        self.client_secret = None


def parse_credential_args(args, options):
    if args.client_id and not args.client_secret:
        raise argparse.ArgumentError(None, "If --client-id is specified, then --client-secret must also be specified.")
    elif args.client_secret and not args.client_id:
        raise argparse.ArgumentError(None, "If --client-secret is specified, then --client-id must also be specified.")
    options.client_id = args.client_id or None
    options.client_secret = args.client_secret or None


def check_commandline_values(args):
    if not args.text and not args.text_file:
        logging.error("Synthesis text and text-file field cannot both be empty")
        raise ValueError("Synthesis text and text-file field cannot both be empty")


def parse_tts_command_line() -> SynthesizerOptions:
    options = SynthesizerOptions()
    parser = argparse.ArgumentParser(description='Perform speech synthesis on a given text')
    parser.add_argument(
        '--voice',
        '-v',
        choices=[
            'tommy_en_us',
            'miguel_es_pe',
            'luz_es_pe',
            'bel_pt_br',
            'david_es_es',
            'anna_ca',
            'arthur_en_us',
            'fiona_en_us',
            'tricia_en_us',
            'marvin_en_us',
            'pablo_es_es',
            'helena_es_es',
            'pedro_pt_br',
            'marcia_pt_br'
        ],
        help='Voice to use for the synthesis',
        required=True)
    parser.add_argument('--sample-rate', '-s', type=int, choices=[8000, 16000], help='Output audio sample rate in Hz', default=16000)
    parser.add_argument('--format', '-f', choices=['wav', 'raw'], help='Output audio format', default='wav')
    parser.add_argument('--audio-file', '-a', help='Path to store the resulting audio', required=True)
    parser.add_argument('--token', '-t', help='File with the authentication token', required=True)
    parser.add_argument('--host', '-H', help='The URL of the host trying to reach', required=True)
    parser.add_argument('--not-secure', '-S', help='Do not use a secure channel. Used for internal testing.',
                        required=False, default=True, dest='secure', action='store_false')
    parser.add_argument('--inactivity-timeout', '-i', help='Time for stream inactivity after the first valid response', required=False, default=5.0)

    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--text', '-T', help='Text to synthesize to audio')
    group.add_argument('--text-file', '-F', help='File with newline delimited text to synthesize to audio')

    credentialGroup = parser.add_argument_group(
        'credentials',
        '''[OPTIONAL] Client authentication credentials used to refresh the token.
        You can find your credentials on the dashboard at https://dashboard.speechcenter.verbio.com/access-token''')
    credentialGroup.add_argument('--client-id', help='Client id for authentication. MUST be written as --client-id=CLIENT_ID')
    credentialGroup.add_argument('--client-secret', help='Client secret for authentication. MUST be written as --client-secret=CLIENT_SECRET')

    args = parser.parse_args()
    check_commandline_values(args)
    parse_credential_args(args, options)

    options.token_file = args.token
    options.host = args.host
    options.audio_file = args.audio_file
    options.secure_channel = args.secure
    options.audio_format = args.format
    options.text = args.text
    options.text_file = args.text_file
    options.voice = args.voice
    options.sample_rate = args.sample_rate
    options.inactivity_timeout = float(args.inactivity_timeout)

    return options


def retrieve_token(options: SynthesizerOptions) -> str:
    if options.client_id:
        return SpeechCenterCredentials.get_refreshed_token(options.client_id, options.client_secret, options.token_file)
    else:
        return SpeechCenterCredentials.read_token(token_file=options.token_file)


class VerbioGrammar:
    INLINE = 0
    URI = 1
    COMPILED = 2

    def __init__(self, grammar_type, content=None):
        self.type = grammar_type
        self.content = content


class RecognizerOptions:
    def __init__(self):
        self.token_file = None
        self.host = ""
        self.audio_file = None
        self.topic = None
        self.grammar = None
        self.language = 'en-US'
        self.secure_channel = True
        self.diarization = False
        self.formatting = False
        self.inactivity_timeout = False
        self.asr_version = None
        self.label = None
        self.client_id = None
        self.client_secret = None

    def check(self):
        if self.topic is None and self.grammar is None:
            raise Exception("You must provide a least a topic or a grammar")
        if self.topic is not None and self.grammar is not None:
            raise Exception("You must provide either a topic or a grammar only, not both")


def parse_csr_commandline() -> RecognizerOptions:
    options = RecognizerOptions()
    parser = argparse.ArgumentParser(description='Perform speech recognition on an audio file')
    parser.add_argument('--audio-file', '-a', help='Path to a .wav audio in 8kHz and PCM16 encoding', required=True)
    parser.add_argument('--convert-audio', '-c', help='Convert audio file to from A-LAW to PCM using sox software. '
                                                      'Used for internal testing.',
                        required=False, default=False, dest='convert_audio', action='store_true')
    topic_group = parser.add_mutually_exclusive_group(required=True)
    topic_group.add_argument('--topic', '-T', choices=['GENERIC', 'TELCO', 'BANKING', 'INSURANCE'],
                             help='A valid topic.')
    topic_group.add_argument('--inline-grammar', '-I', help='Grammar inline as a string.')
    topic_group.add_argument('--grammar-uri', '-G', help='Builtin grammar URI for the recognition.')
    topic_group.add_argument('--compiled-grammar', '-C', help='The compiled grammar file path (an .tar.xz) for the recognition.')
    parser.add_argument(
        '--language',
        '-l',
        choices=[
            'en',
            'en-US',
            'en-GB',
            'pt-BR',
            'es',
            'ca-ES',
            'es-419',
            'tr',
            'ja',
            'fr',
            'fr-CA',
            'de',
            'it'],
        help='A Language ID (default: ' + options.language + ')',
        default=options.language)
    parser.add_argument('--token', '-t', help='File with the authentication token', required=True)
    parser.add_argument('--host', '-H', help='The URL of the host trying to reach (default: ' + options.host + ')',
                        required=True)
    parser.add_argument('--not-secure', '-S', help='Do not use a secure channel. Used for internal testing.',
                        required=False, default=True, dest='secure', action='store_false')
    parser.add_argument('--diarization', '-d', help='', required=False, default=False, action='store_true')
    parser.add_argument('--formatting', '-f', help='', required=False, default=False, action='store_true')
    parser.add_argument('--inactivity-timeout', '-i', help='Time for stream inactivity after the first valid response',
                        required=False, default=5.0)
    parser.add_argument('--asr-version', choices=['V1', 'V2'], help='Selectable asr version', required=True)
    parser.add_argument('--label', help='Label for the request', required=False, default="")

    credential_group = parser.add_argument_group(
        'credentials',
        '''[OPTIONAL] Client authentication credentials used to refresh the token.
        You can find your credentials on the dashboard at https://dashboard.speechcenter.verbio.com/access-token''')
    credential_group.add_argument('--client-id',
                                  help='Client id for authentication. MUST be written as --client-id=CLIENT_ID')
    credential_group.add_argument('--client-secret',
                                  help='Client secret for authentication. MUST be written as --client-secret=CLIENT_SECRET')

    args = parser.parse_args()
    parse_credential_args(args, options)

    options.token_file = args.token
    options.host = args.host
    options.audio_file = args.audio_file
    options.convert_audio = args.convert_audio
    options.language = args.language
    options.secure_channel = args.secure
    options.formatting = args.formatting
    options.diarization = args.diarization
    options.inactivity_timeout = float(args.inactivity_timeout)
    options.asr_version = args.asr_version
    options.label = args.label

    if args.inline_grammar:
        options.grammar = VerbioGrammar(VerbioGrammar.INLINE, args.inline_grammar)
    elif args.compiled_grammar:
        options.grammar = VerbioGrammar(VerbioGrammar.COMPILED, args.compiled_grammar)
    elif args.grammar_uri:
        options.grammar = VerbioGrammar(VerbioGrammar.URI, args.grammar_uri)
    else:  # No grammars
        options.topic = args.topic

    return options


def split_audio(audio: bytes, chunk_size: int = 20000):
    audio_length = len(audio)
    chunk_count = math.ceil(audio_length / chunk_size)
    logging.info("Dividing audio of length " + str(audio_length) + " into " + str(chunk_count) + " of size " + str(chunk_size) + "...")
    if chunk_count > 1:
        for i in range(chunk_count):
            start = i * chunk_size
            end = min((i + 1) * chunk_size, audio_length)
            logging.info("Audio chunk #" + str(i) + " sliced as " + str(start) + ":" + str(end))
            yield audio[start:end]
    else:
        yield audio


def split_text(text_file: str):
    with open(text_file) as f:
        for (i, line) in enumerate(f):
            text = line.rstrip()
            logging.info("Text slice #" + str(i) + ": " + text)
            yield text
