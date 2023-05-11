#!/usr/bin/env python3
import grpc


class GrpcChannelCredentials:
    def __init__(self, token):
        # Set JWT token for service access.
        self.call_credentials = grpc.access_token_call_credentials(token)
        # Set CA Certificate for SSL channel encryption.
        self.ssl_credentials = grpc.ssl_channel_credentials()

    def get_channel_credentials(self):
        return grpc.composite_channel_credentials(self.ssl_credentials, self.call_credentials)
