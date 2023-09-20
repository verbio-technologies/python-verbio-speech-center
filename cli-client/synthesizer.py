import verbio_speech_center_synthesizer_pb2_grpc
from helpers.common import SynthesizerOptions, parse_tts_command_line, retrieve_token
from helpers.grpc_connection import GrpcConnection
from helpers.audio_exporter import AudioExporter
from helpers.tts_client import TTSClient
import logging
import sys
sys.path.insert(1, '../proto/generated')


def run(options: SynthesizerOptions):
    access_token = retrieve_token(command_line_options)
    grpc_connection = GrpcConnection(options.secure_channel, options.client_id, options.client_secret, access_token)

    with grpc_connection.open(options.host) as grpc_channel:
        grpc_stub = verbio_speech_center_synthesizer_pb2_grpc.TextToSpeechStub(grpc_channel)
        client = TTSClient(grpc_stub, options, access_token)
        audio_samples = client.synthesize(options.text, options.voice)
        logging.info("Received response with %s bytes of audio data", len(audio_samples))

        audio_exporter = AudioExporter(options.sample_rate)
        audio_exporter.save_audio(options.audio_format, audio_samples, options.audio_file)
        logging.info("Stored synthesis audio at -%s-. Synthesis finished.", options.audio_file)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s][%(levelname)s]:%(message)s')
    logging.info("Running speechcenter TTS synthesis...")
    command_line_options = parse_tts_command_line()
    run(command_line_options)
