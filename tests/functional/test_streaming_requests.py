import time
from .recognizer_service_test_case import RecognizerServiceTestCase


class TestStreamingEnglish(RecognizerServiceTestCase):
    async def testEnglish(self):
        idx, mergedResponse = 0, None
        expectedResponses = [
            "hey enzo josh whitekid's here calling from salesforce hope that you enjoyed the warriors game last week um and they won last night which is great i wanted",
            "to uh I'm calling for two things one to talk to you about uh your marketing initiatives around social we have uh we just released a new version of our social studio",
            "which helps to monitor brand respond and manage multiple brands there um and also i wanted to talk to you about a marketing assessment that we offer as a free",
            "free service we look at your strategy we look at uh what solutions and data you're you're using your operations and and how you're interacting with customers and",
            "the whole life cycle and then we put uh together some tailored recommendations so um couple couple great activities that we could do as some next steps here wanted to figure out who",
            "the right people on your team are to calendar this with give me a call back six five oh six five three two nine nine two thank you",
        ]
        async for response in self.requestAsync("en-us-2.wav", "en-US"):
            # Latency: avg(wordLatencies) ~ (maxLatency + minLatency) / 2 = (5 seconds + processingTime) where:
            # maxLatency = timeForinputBufferToFill (10 seconds) + processingTime
            # minLatency = processingTime
            self.expectLatency(response, time.time(), 6.0)
            self.expectNotEmptyTranscription(response)
            self.expectNumberOfWords(response, len(expectedResponses[idx].split()), 3)

            isLastResponse = not idx < len(expectedResponses) - 1
            audioChunkDurationSeconds = 7 if isLastResponse else 10
            audioDurationSeconds = 57 if isLastResponse else 10 * (idx + 1)
            audioDurationNanos = 960000000 if isLastResponse else 000000000
            mergedResponse = RecognizerServiceTestCase.mergeResponsesIntoOne(
                mergedResponse, response
            )

            self.expectValidConfidence(response.results.alternatives[0].confidence)
            self.expectDuration(
                response.results.duration,
                seconds=audioChunkDurationSeconds,
                nanos=audioDurationNanos,
            )
            self.expectDuration(
                response.results.end_time,
                seconds=audioDurationSeconds,
                nanos=audioDurationNanos,
            )
            self.expectFinal(response)
            self.expectValidWords(response)
            self.expectValidWordTimestamps(
                response,
                audioDuration=float(f"{audioDurationSeconds}.{audioDurationNanos}"),
            )

            idx += 1

        self.expectTranscriptionWER(
            mergedResponse, " ".join(expectedResponses), 0.16374269005847952, delta=2e-2
        )
        self.expectTranscriptionCER(
            mergedResponse, " ".join(expectedResponses), 0.09745293466223699, delta=2e-2
        )
        self.expectDuration(
            mergedResponse.results.duration, seconds=57, nanos=960000000
        )
        self.expectDuration(
            mergedResponse.results.end_time, seconds=57, nanos=960000000
        )
        self.expectFinal(mergedResponse)
        self.expectValidWords(mergedResponse)
        self.expectValidWordTimestamps(mergedResponse, audioDuration=57.960000000)


class TestStreamingSpanish(RecognizerServiceTestCase):
    _language = "es"

    async def testSpanish(self):
        idx, mergedResponse = 0, None
        expectedResponses = [
            "el otro día me ha hecho oferta no",
            "no no me interesa venga suelta",
            "jazztel dos móviles",
            "fijo de todo y todo",
            "sí sí",
            "",
        ]
        async for response in self.requestAsync("es-2.wav", "es"):
            self.expectLatency(response, time.time(), 8.0)

            isLastResponse = not idx < len(expectedResponses) - 1
            if not isLastResponse:
                self.expectNotEmptyTranscription(response)

            self.expectNumberOfWords(response, len(expectedResponses[idx].split()), 3)

            audioChunkDurationSeconds = 2 if isLastResponse else 10
            audioDurationSeconds = 52 if isLastResponse else 10 * (idx + 1)
            mergedResponse = RecognizerServiceTestCase.mergeResponsesIntoOne(
                mergedResponse, response
            )

            self.expectValidConfidence(response.results.alternatives[0].confidence)
            self.expectDuration(
                response.results.duration, seconds=audioChunkDurationSeconds, nanos=0
            )
            self.expectDuration(
                response.results.end_time, seconds=audioDurationSeconds, nanos=0
            )
            self.expectFinal(response)
            self.expectValidWords(response)
            self.expectValidWordTimestamps(
                response, audioDuration=float(f"{audioDurationSeconds}.000000000")
            )

            idx += 1

        self.expectTranscriptionWER(
            mergedResponse, " ".join(expectedResponses), 0.5217391304347826, delta=5e-2
        )
        self.expectTranscriptionCER(
            mergedResponse, " ".join(expectedResponses), 0.37962962962962965, delta=2e-2
        )
        self.expectDuration(
            mergedResponse.results.duration, seconds=52, nanos=000000000
        )
        self.expectDuration(
            mergedResponse.results.end_time, seconds=52, nanos=000000000
        )
        self.expectFinal(mergedResponse)
        self.expectValidWords(mergedResponse)
        self.expectValidWordTimestamps(mergedResponse, audioDuration=52.0)


class TestStreamingPortuguese(RecognizerServiceTestCase):
    _language = "pt-br"

    async def testPortuguese(self):
        idx, mergedResponse = 0, None
        expectedResponses = [
            "beleza então não tem problema não tá certo eu tenho uma informação aqui pra passar pro senhor o senhor pode ouvir diga o banco concedeu",
            "da seguradora a proteção e a proteção do cartão o cartão protegido plus no valor de apenas sete e noventa e nove mensal pro cartão antigo",
            "daí é possível senhor josé o senhor fazer essa contratação e por uma cobertura de até vinte mil para saque e transações e compras efetuadas sobre inflação",
            "com cartão segurado também o senhor recupera os bens roubados juntamente com pasta mochila sacola contendo o cartão segurado mediante",
            "envio da nota fiscal indenização no valor de até três mil reais e tem também proteção de preço algo muito legal se o senhor comprar alguma coisa e em trinta dias aquele mesmo produto",
            "cair o valor o banco ele te dá uma indenização no valor de até mil reais além de concorrer a sorteios na loteria federal no valor de dez mil reais",
        ]
        async for response in self.requestAsync("pt-br-2.wav", "pt-BR"):
            self.expectLatency(response, time.time(), 7.0)
            self.expectNotEmptyTranscription(response)
            self.expectNumberOfWords(response, len(expectedResponses[idx].split()), 3)

            mergedResponse = RecognizerServiceTestCase.mergeResponsesIntoOne(
                mergedResponse, response
            )

            self.expectValidConfidence(response.results.alternatives[0].confidence)
            self.expectDuration(response.results.duration, seconds=10, nanos=000000000)
            self.expectDuration(
                response.results.end_time, seconds=10 * (idx + 1), nanos=000000000
            )
            self.expectFinal(response)
            self.expectValidWords(response)
            self.expectValidWordTimestamps(response, audioDuration=10 * (idx + 1))

            idx += 1

        self.expectTranscriptionWER(
            mergedResponse, " ".join(expectedResponses), 0.24074074074074073, delta=3e-2
        )
        self.expectTranscriptionCER(
            mergedResponse, " ".join(expectedResponses), 0.1244343891402715, delta=2e-2
        )
        self.expectDuration(mergedResponse.results.duration, seconds=60, nanos=0)
        self.expectDuration(mergedResponse.results.end_time, seconds=60, nanos=0)
        self.expectFinal(mergedResponse)
        self.expectValidWords(mergedResponse)
        self.expectValidWordTimestamps(mergedResponse, audioDuration=60.0)
