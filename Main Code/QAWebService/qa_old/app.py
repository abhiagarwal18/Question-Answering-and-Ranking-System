from flask import Flask, request, Response,jsonify
import requests
from deeppavlov import build_model, configs
import os
from google.cloud import translate
from pygame import mixer

# set google authentication
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./myrpproject1-bdf39beed338.json"
model = build_model(configs.squad.squad, download=True)

# initialization
mixer.init()
translate_client = translate.Client()

context_list = []
context = ' Bits Pilani is a private institute of higher education and a deemed university under Section 3 of the UGC Act 1956. \
The institute was established in its present form in 1964. It is established across 4 campuses and has 15 academic departments. \
Pilani is located 220 kilometres far from Delhi in Rajasthan. We can reach bits pilani via train or bus from delhi. \
Bits Pilani has its campuses in Pilani , Goa , Hyderabad , Dubai. There are multiple scholarships available at BITS namely Merit Scholarships, Merit Cum Need Scholarships and BITSAA Scholarships. \
BITS Model United Nations Conference (BITSMUN) is one of the largest MUN conferences in the country. BITS conducts the All-India computerized entrance examination, BITSAT (BITS Admission Test). \
Admission is merit-based, as assessed by the BITSAT examination. \
We can reach bits pilani through bus or train from delhi or jaipur. \
Mr. Ashoke Kumar Sarkar is the director of Bits Pilani, pilani campus. \
Founder of Bits pilani was Ghanshyam Das Birla.'

context_list.append(context)

app = Flask(__name__)
app.debug = True

@app.route("/check")
def checkServer():
	return "up and running"

@app.route("/chat", methods = ["POST"])
def getAnswerService():
	body_response = {}
	
	question = request.get_json(force=True)["message"]
	contextId = request.get_json(force=True)["topicId"]
	language = request.get_json(force=True)["language"]
	print(question)
	if language != 'en':
		en_question=translate_client.translate(question,target_language='en')['translatedText']
	else:
		en_question=question

	en_question_list = []
	en_question_list.append(en_question)
	answer = model(context_list,en_question_list)
	en_reply = answer[0][0]
	
	lang_reply=translate_client.translate(en_reply,target_language=language)['translatedText']

	body_response["reply"] = lang_reply
	print(lang_reply,'\n')
	return jsonify(body_response)

@app.route("/topics", methods = ["GET"])
def getTopicsService():
	body_response=[]
	body_response_item = {}
	body_response_item['id']=1
	body_response_item['name']='Bits Pilani'
	body_response_item['description']=context
	body_response.append(body_response_item)
	return jsonify(body_response)
	
if(__name__ == "__main__"):
	app.run(host='0.0.0.0',port = "5000")
