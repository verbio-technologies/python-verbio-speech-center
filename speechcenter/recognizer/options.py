#!/usr/bin/env python3
import logging


class Options:
    def __init__(self, args):
        self.token_file = args.token
        self.host = args.host
        self.audio_file = args.audio_file
        self.topic = args.topic
        self.language = args.language or 'en-US'
        self.secure_channel = args.secure
        self.formatting = args.formatting
        self.diarization = args.diarization
        self.inactivity_timeout = float(args.inactivity_timeout)
        self.asr_version = args.asr_version
        self.label = args.label
        self.logging_level = logging.getLevelName(args.logging_level)

        self.client_id, self.client_secret = self.check_credentials(args)

        self.asr_version = self.check_asr_version(args.asr_version)
    
    def check_credentials(self, args):
        if args.client_id and not args.client_secret:
            raise Exception("If --client-id is specified, then --client-secret must also be specified.")
        elif args.client_secret and not args.client_id:
            raise Exception("If --client-secret is specified, then --client-id must also be specified.")
        
        return (args.client_id or None, args.client_secret or None)
    
    def check_asr_version(self, asr_version):
        if len(asr_version):
            asr_versions = {"V1":0, "V2":1}
            return asr_versions[asr_version]
        else:
            raise Exception("ASR version must be declared in order to perform the recognition.")

    def check(self):
        if self.topic is None:
            raise Exception("You must provide a least a topic.")
