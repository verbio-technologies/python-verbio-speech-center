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
        self.text: str = None
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
    if not args.text:
        logging.error("Synthesis text field needs to be non empty")
        raise ValueError("Synthesis text field needs to be non empty")


def parse_tts_command_line() -> SynthesizerOptions:
    options = SynthesizerOptions()
    parser = argparse.ArgumentParser(description='Perform speech synthesis on a given text')
    parser.add_argument('--text', '-T', help='Text to synthesize to audio', required=True)
    parser.add_argument(
        '--voice',
        '-v',
        choices=[
            'tommy_en_us',
            'miguel_es_pe',
            'bel_pt_br',
            'david_es_es',
            'anna_ca'],
        help='Voice to use for the synthesis',
        required=True)
    parser.add_argument('--sample-rate', '-s', type=int, choices=[16000], help='Output audio sample rate in Hz', default=16000)
    parser.add_argument('--format', '-f', choices=['wav', 'raw'], help='Output audio format', default='wav')
    parser.add_argument('--audio-file', '-a', help='Path to store the resulting audio', required=True)
    parser.add_argument('--token', '-t', help='File with the authentication token', required=True)
    parser.add_argument('--host', '-H', help='The URL of the host trying to reach', required=True)
    parser.add_argument('--not-secure', '-S', help='Do not use a secure channel. Used for internal testing.',
                        required=False, default=True, dest='secure', action='store_false')

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
    options.voice = args.voice
    options.sample_rate = args.sample_rate

    return options


def retrieve_token(options: SynthesizerOptions) -> str:
    if options.client_id:
        return SpeechCenterCredentials.get_refreshed_token(options.client_id, options.client_secret, options.token_file)
    else:
        return SpeechCenterCredentials.read_token(token_file=options.token_file)


class RecognizerOptions:
    def __init__(self):
        self.token_file = None
        self.host = ""
        self.audio_file = None
        self.topic = None
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
        if self.topic is None:
            raise Exception("You must provide a least a topic")



def parse_csr_commandline() -> RecognizerOptions:
    options = RecognizerOptions()
    parser = argparse.ArgumentParser(description='Perform speech recognition on an audio file')
    parser.add_argument('--audio-file', '-a', help='Path to a .wav audio in 8kHz and PCM16 encoding', required=True)
    topicGroup = parser.add_mutually_exclusive_group(required=True)
    topicGroup.add_argument('--topic', '-T', choices=['GENERIC', 'TELCO', 'BANKING', 'INSURANCE'], help='A valid topic')
    parser.add_argument(
        '--language',
        '-l',
        choices=[
            'en',
            'en-US',
            'en-GB',
            'pt-BR',
            'es',
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
    parser.add_argument('--host', '-H', help='The URL of the host trying to reach (default: ' + options.host + ')', required=True)
    parser.add_argument('--not-secure', '-S', help='Do not use a secure channel. Used for internal testing.',
                        required=False, default=True, dest='secure', action='store_false')
    parser.add_argument('--diarization', '-d', help='', required=False, default=False, action='store_false')
    parser.add_argument('--formatting', '-f', help='', required=False, default=False, action='store_false')
    parser.add_argument('--inactivity-timeout', '-i', help='Time for stream inactivity after the first valid response', required=False, default=5.0)
    parser.add_argument('--asr-version', choices=['V1', 'V2'], help='Selectable asr version', required=True)
    parser.add_argument('--label', help='"Label for the request', required=False, default="")

    credentialGroup = parser.add_argument_group(
        'credentials',
        '''[OPTIONAL] Client authentication credentials used to refresh the token.
        You can find your credentials on the dashboard at https://dashboard.speechcenter.verbio.com/access-token''')
    credentialGroup.add_argument('--client-id', help='Client id for authentication. MUST be written as --client-id=CLIENT_ID')
    credentialGroup.add_argument('--client-secret', help='Client secret for authentication. MUST be written as --client-secret=CLIENT_SECRET')

    args = parser.parse_args()
    parse_credential_args(args, options)

    options.token_file = args.token
    options.host = args.host
    options.audio_file = args.audio_file
    options.topic = args.topic
    options.language = args.language
    options.secure_channel = args.secure
    options.formatting = args.formatting
    options.diarization = args.diarization
    options.inactivity_timeout = float(args.inactivity_timeout)
    options.asr_version = args.asr_version
    options.label = args.label

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
