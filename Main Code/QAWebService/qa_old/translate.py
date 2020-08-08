#import requests
#import urllib3
#urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from deeppavlov import build_model, configs
import os
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "/home/kd/Workspace/translate/myrpproject1-bdf39beed338.json"



model = build_model(configs.squad.squad, download=True)

context = []
cstring = 'Bits Pilani is a private institute of higher education and a deemed university under Section 3 of the UGC Act 1956.\n\
    The institute was established in its present form in 1964.\n\
    It is established across 4 campuses and has 15 academic departments.\n\
    Pilani is located 220 kilometres far from Delhi.\n\
    Bits Pilani has its campuses in Pilani , Goa , Hyderabad , Dubai .'
print(cstring)
context.append(cstring)



question = []
qstring = 'what is bits pilani?'
question.append(qstring)

while(1):
	answer = model(context,question)

	print(answer[0][0])
