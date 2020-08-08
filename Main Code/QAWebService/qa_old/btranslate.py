import speech_recognition as sr
from translation import google
from google.cloud import translate
import requests
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
import pyaudio
import wave
from deeppavlov import build_model, configs
from pygame import mixer
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/kd/Workspace/translate/myrpproject1-bdf39beed338.json"

mixer.init()



translate_client = translate.Client()

model = build_model(configs.squad.squad, download=True)

while (True == True):
# obtain audio from the microphone
  r = sr.Recognizer()
  with sr.Microphone() as source:
    #print("Please wait. Calibrating microphone...")
    # listen for 1 second and create the ambient noise energy level
    r.adjust_for_ambient_noise(source, duration=1)
    print("Here is the Context. Please Ask A Question")
    print(" ")
    print("Context - ")
    print(" ")
    context = []
    cstring = 'Bits Pilani is a private institute of higher education and a deemed university under Section 3 of the UGC Act 1956.\n\
    The institute was established in its present form in 1964.\n\
    It is established across 4 campuses and has 15 academic departments.\n\
    Pilani is located 220 kilometres far from Delhi.\n\
    Bits Pilani has its campuses in Pilani , Goa , Hyderabad , Dubai .'
    print(cstring)
    context.append(cstring)
    print(" ")
    print("Ask Any Question")
    audio = r.listen(source,phrase_time_limit=5)


# recognize speech using Sphinx/Google
  try:
    text = r.recognize_google(audio,language = 'hi-IN')
    print("I think you said '" + text + "'")
    target = 'en'
    translation = translate_client.translate(
        text,
        target_language=target)

    print(u'Text: {}'.format(text))
    print(u'Translation: {}'.format(translation['translatedText']))

    if translation['translatedText'] == 'bye':
        break
    # Model Code Starts



    #context = []
    #cstring = 'Qutub Minar is in Delhi. We can reach Qutub Minar By Metro. Narendra Modi is the prime minister of india.'
    #print("Here is your Context. Please ask a Question")
    question = []
    qstring = translation['translatedText']
    question.append(qstring)

    answer = model(context,question)

    print(answer[0])
    # End
    target = 'hi'
    translation = translate_client.translate(
        answer[0][0],
        target_language=target)
    print(u'Translation: {}'.format(translation['translatedText']))

    u1 = "http://ivrapi.indiantts.co.in/tts?type=indiantts&text="
    umid = translation['translatedText']
    u2 = "&api_key=2d108780-0b86-11e6-b056-07d516fb06e1&user_id=80&action=play"
    url = u1+umid+u2
    #print(url)
    req = requests.get(url)
    #print(req.content)


    with open("sound.wav",'wb') as f:
        f.write(req.content)


    #define stream chunk
    chunk = 1024

    #open a wav format music
    f = wave.open("sound.wav","rb")
    #instantiate PyAudio
    p = pyaudio.PyAudio()
    #open stream
    stream = p.open(format = p.get_format_from_width(f.getsampwidth()),
                    channels = f.getnchannels(),
                    rate = f.getframerate(),
                    output = True)
    #read data
    data = f.readframes(chunk)

    #play stream
    while data:
        stream.write(data)
        data = f.readframes(chunk)

    #stop stream
    stream.stop_stream()
    stream.close()

    #close PyAudio
    p.terminate()

  except sr.UnknownValueError:
    print("Sphinx could not understand audio")
  except sr.RequestError as e:
    print("Sphinx error; {0}".format(e))
