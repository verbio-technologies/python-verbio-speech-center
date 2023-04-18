
import jwt
import requests
from datetime import datetime
import json
import logging


class SpeechCenterCredentials:
        
    @staticmethod
    def read_token(token_file: str) -> str:
        with open(token_file) as token_hdl:
            return ''.join(token_hdl.read().splitlines())

    @staticmethod
    def get_refreshed_token(client_id, client_secret, token_file):
        currentToken = SpeechCenterCredentials.read_token(token_file=token_file)
        refresh = False
        try:
            payload = jwt.decode(currentToken, options={"verify_signature": False})
            if datetime.now().timestamp() >= payload['exp']:
                logging.info("Provided token is expired, proceeding to refresh")
                refresh = True
        except Exception:
            logging.info("Provided file does not contain a valid JWT token, proceeding to retrieve a new token")
            refresh = True
        if refresh:
            newToken = SpeechCenterCredentials._refresh_token(client_id, client_secret)
            SpeechCenterCredentials._writeNewToken(newToken, token_file)
            return newToken
        else:
            logging.info("Provided token is still valid, skipping refresh")
            return currentToken
    
    @staticmethod
    def _refresh_token(client_id, client_secret):

        headers = {'Accept': 'application/json',
                   'Content-Type': 'application/json'}
        body = """{
        "client_id":"%s",
        "client_secret": "%s"
        }""" % (client_id, client_secret)

        response = requests.post("https://auth.speechcenter.verbio.com:444/api/v1/token", headers=headers, data=body)
        parsedResponse = json.loads(response.content)

        if response.status_code != 200:
            raise ConnectionRefusedError("Cannot refresh token. Error: " + parsedResponse['error'] + ": " + parsedResponse['message'])
        else:
            logging.info("Succesfully updated service token:\n" + parsedResponse['access_token'])
            logging.info("New expiration time is:\n" + str(parsedResponse['expiration_time']))
        

        return parsedResponse['access_token']

    @staticmethod
    def _writeNewToken(token, file):
        with open(file, "w") as tokenFile:
            tokenFile.write(token)