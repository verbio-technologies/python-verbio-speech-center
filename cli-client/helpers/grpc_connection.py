import grpc
import logging


class GrpcChannelCredentials:
    def __init__(self, token):
        self.call_credentials = grpc.access_token_call_credentials(token)
        self.ssl_credentials = grpc.ssl_channel_credentials()

    def get_channel_credentials(self):
        return grpc.composite_channel_credentials(self.ssl_credentials, self.call_credentials)


class GrpcConnection:
    def __init__(self, secure: bool, client_id: str, client_secret: str, access_token: str):
        self._client_id = client_id
        self._client_secret = client_secret
        self._access_token = access_token
        self._secure = secure

    def open(self, host: str):
        if self._secure:
            logging.info("Connecting to %s using a secure channel...", host)
            credentials = GrpcChannelCredentials(self._access_token)
            return grpc.secure_channel(host, credentials=credentials.get_channel_credentials())
        else:
            logging.info("Connecting to %s using a insecure channel...", host)
            return grpc.insecure_channel(host)

