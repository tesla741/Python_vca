import json
from os.path import join, dirname
from ibm_watson import SpeechToTextV1
from ibm_watson.websocket import RecognizeCallback, AudioSource
from ibm_cloud_sdk_core.authenticators import IAMAuthenticator


file_name=''

authenticator = IAMAuthenticator('Hcv70i0aU1-MzuidFoEMas04DOjb-MQjkXr6a0eAMe8j')
speech_to_text = SpeechToTextV1(
    authenticator=authenticator
)



speech_to_text.set_service_url('https://gateway-lon.watsonplatform.net/speech-to-text/api')

class MyRecognizeCallback(RecognizeCallback):
    def __init__(self):
        RecognizeCallback.__init__(self)

    def on_data(self, data):
        print(json.dumps(data, indent=2))
        f = open('Watson1.json', "w+")
        f.write(json.dumps(data, indent=2))
        f.close()
        

    def on_error(self, error):
        print('Error received: {}'.format(error))

    def on_inactivity_timeout(self, error):
        print('Inactivity timeout: {}'.format(error))

myRecognizeCallback = MyRecognizeCallback()
file_path = '/Users/aarooran.v/Documents/Vca_v1/Voice Recordings/'

def fileName_transcript(file_name):
    
    with open(join(dirname(file_path), file_name),
                'rb') as audio_file:
        audio_source = AudioSource(audio_file)
        speech_to_text.recognize_using_websocket(
            audio=audio_source,
            content_type='audio/flac',
            speaker_labels=True,
            recognize_callback=myRecognizeCallback,
            model='en-US_BroadbandModel',
            keywords=['hold', 'Sean', 'job'],
            keywords_threshold=0.5,
            # max_alternatives=3
            )


# fileName_transcript('Scenario1Compliant.flac')