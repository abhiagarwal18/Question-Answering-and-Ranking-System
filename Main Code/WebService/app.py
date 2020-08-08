from flask import Flask, request, Response, jsonify, render_template
import flask
import requests
import urllib.request
import time
from bs4 import BeautifulSoup
from urllib.request import urlopen
import csv,json

# Cosine Similarity
import nltk, string, numpy

from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer

import pandas as pd
import numpy as np
import re, nltk
import gensim
import codecs
import pickle
import en_core_web_sm
from sner import Ner
import spacy
from sklearn.metrics import confusion_matrix, accuracy_score, average_precision_score
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, GridSearchCV
from nltk.internals import find_jars_within_path
from nltk.tag import StanfordPOSTagger
from nltk.tag import StanfordNERTagger
from sklearn import linear_model
from sklearn import svm
from sklearn.metrics import fbeta_score, accuracy_score
from scipy.sparse import hstack

from nltk.stem.porter import PorterStemmer 
from nltk.stem.snowball import SnowballStemmer
from nltk.stem.wordnet import WordNetLemmatizer

import requests

from bert import QA

model = QA('model')

context = ' Bits Pilani is a private institute of higher education and a deemed university under Section 3 of the UGC Act 1956. \
The institute was established in its present form in 1964. It is established across 4 campuses and has 15 academic departments. \
Pilani is located 220 kilometres far from Delhi in Rajasthan. We can reach bits pilani via train or bus from delhi. \
Bits Pilani has its campuses in Pilani , Goa , Hyderabad , Dubai. There are multiple scholarships available at BITS namely Merit Scholarships, Merit Cum Need Scholarships and BITSAA Scholarships. \
BITS Model United Nations Conference (BITSMUN) is one of the largest MUN conferences in the country. BITS conducts the All-India computerized entrance examination, BITSAT (BITS Admission Test). \
Admission is merit-based, as assessed by the BITSAT examination. \
We can reach bits pilani through bus or train from delhi or jaipur. \
Mr. Ashoke Kumar Sarkar is the director of Bits Pilani, pilani campus. \
Founder of Bits pilani was Ghanshyam Das Birla.'

def get_answer(context:str, ques:str):
    answer= model.predict(context, ques)
    return answer['answer']


app = Flask(__name__)
app.debug = True

QALinks = {
        '0':"context",
		'1':"demo",
		'2':"demo",
        '3':"demo",
		'4':"demo",
        '5':"demo"
	}

@app.route("/check")
def checkServer():
	return "up and running"

@app.route("/search", methods = ["GET","POST"])
def srchLinks():
    questionString = str(request.form["question"])
 
    bits = 0
    response_data={}
    for token in questionString.split():
        token = token.lower()
        if token == 'bits' or token == 'pilani':
            bits = 1

    if bits == 1:
        predicted_class = "BITS Pilani"
        QALinks['0']=predicted_class
        QALinks['1']=get_answer(context, questionString)
        QALinks.pop('2', None)
        QALinks.pop('3', None)
        QALinks.pop('4', None)
        QALinks.pop('5', None)

        # print(response_data)
        response = flask.jsonify(QALinks)
        response.headers.add('Access-Control-Allow-Origin', '*')
        return response

    #creating question
    qstn = questionString.replace(' ','+')
    
    questions = []  # a list to store link of questions 
    documents = []  # # a list to store questions in words
    
    documents.append(questionString)
    questions.append('Original question')



	#Context Identification of Questions

    def text_clean(corpus, keep_list):
        '''
        Purpose : Function to keep only alphabets, digits and certain words (punctuations, qmarks, tabs etc. removed)
        
        Input : Takes a text corpus, 'corpus' to be cleaned along with a list of words, 'keep_list', which have to be retained
                even after the cleaning process
        
        Output : Returns the cleaned text corpus
        
        '''
        cleaned_corpus = pd.Series()
        for row in corpus:
            qs = []
            for word in row.split():
                if word not in keep_list:
                    p1 = re.sub(pattern='[^a-zA-Z0-9]',repl=' ',string=word)
                    p1 = p1.lower()
                    qs.append(p1)
                else : qs.append(word)
            cleaned_corpus = cleaned_corpus.append(pd.Series(' '.join(qs)))
        return cleaned_corpus

    def preprocess(corpus, keep_list, cleaning = True, stemming = False, stem_type = None, lemmatization = False, remove_stopwords = True):  
        '''
        Purpose : Function to perform all pre-processing tasks (cleaning, stemming, lemmatization, stopwords removal etc.)
        
        Input : 
        'corpus' - Text corpus on which pre-processing tasks will be performed
        'keep_list' - List of words to be retained during cleaning process
        'cleaning', 'stemming', 'lemmatization', 'remove_stopwords' - Boolean variables indicating whether a particular task should 
                                                                    be performed or not
        'stem_type' - Choose between Porter stemmer or Snowball(Porter2) stemmer. Default is "None", which corresponds to Porter
                    Stemmer. 'snowball' corresponds to Snowball Stemmer
        
        Note : Either stemming or lemmatization should be used. There's no benefit of using both of them together
        
        Output : Returns the processed text corpus
        
        '''
        if cleaning == True:
            corpus = text_clean(corpus, keep_list)
        
        if remove_stopwords == True:
            wh_words = ['who', 'what', 'when', 'why', 'how', 'which', 'where', 'whom']
            stop = set(stopwords.words('english'))
            for word in wh_words:
                stop.remove(word)
            corpus = [[x for x in x.split() if x not in stop] for x in corpus]
        else :
            corpus = [[x for x in x.split()] for x in corpus]
        
        if lemmatization == True:
            lem = WordNetLemmatizer()
            corpus = [[lem.lemmatize(x, pos = 'v') for x in x] for x in corpus]
        
        if stemming == True:
            if stem_type == 'snowball':
                stemmer = SnowballStemmer(language = 'english')
                corpus = [[stemmer.stem(x) for x in x] for x in corpus]
            else :
                stemmer = PorterStemmer()
                corpus = [[stemmer.stem(x) for x in x] for x in corpus]
        
        corpus = [' '.join(x) for x in corpus]
        return corpus

    common_dot_words = ['U.S.', 'St.', 'Mr.', 'Mrs.', 'D.C.']
    nlp = en_core_web_sm.load()


    # read information from file and load the model
    count_vec_ner = pickle.load(open("count_vec_ner.p", "rb"))
    count_vec_dep = pickle.load(open("count_vec_dep.p", "rb"))
    count_vec_lemma = pickle.load(open("count_vec_lemma.p", "rb"))
    count_vec_shape = pickle.load(open("count_vec_shape.p", "rb"))
    count_vec_tag = pickle.load(open("count_vec_tag.p", "rb"))

    with open('my_question_classifier.pkl', 'rb') as fid:
        model = pickle.load(fid)

    
    classes = {
        "0": "BITS Pilani",
        "1": "Society & Culture",
        "2": "Science & Mathematics",
        "3": "Health",
        "4": "Education & Reference",
        "5": "Computers & Internet",
        "6": "Sports",
        "7": "Business & Finance",
        "8": "Entertainment & Music",
        "9": "Family & Relationships",
        "10": "Politics & Government"
    }

    myQ = {
        'Question': questionString
    }

    bits = 0

    for token in myQ['Question'].split():
        token = token.lower()
        if token == 'bits' or token == 'pilani':
            bits = 1

    if bits == 1:
        predicted_class = classes["0"]
        print(predicted_class)

    else:
        ques = pd.Series(myQ['Question']).astype(str)

        ques = preprocess(ques, keep_list = common_dot_words, remove_stopwords = True)

        myQ_ner = []
        myQ_lemma = []
        myQ_tag = []
        myQ_dep = []
        myQ_shape = []

        doc = nlp(ques[0])
        present_lemma = []
        present_tag = []
        present_dep = []
        present_shape = []
        present_ner = []

        for token in doc:
            present_lemma.append(token.lemma_)
            present_tag.append(token.tag_)
            present_dep.append(token.dep_)
            present_shape.append(token.shape_)
            
        myQ_lemma.append(" ".join(present_lemma))
        myQ_tag.append(" ".join(present_tag))
        myQ_dep.append(" ".join(present_dep))
        myQ_shape.append(" ".join(present_shape))

        # Named entities are available as the ents property of a Doc
        doc = nlp(myQ['Question'])
        for ent in doc.ents:
            present_ner.append(ent.label_)
        myQ_ner.append(" ".join(present_ner))

        ner_myQ_ft = count_vec_ner.transform(myQ_ner)
        lemma_myQ_ft = count_vec_lemma.transform(myQ_lemma)
        tag_myQ_ft = count_vec_tag.transform(myQ_tag)
        dep_myQ_ft = count_vec_dep.transform(myQ_dep)
        shape_myQ_ft = count_vec_shape.transform(myQ_shape)

        x_all_ft_myQ = hstack([ner_myQ_ft, lemma_myQ_ft, tag_myQ_ft])

        x_all_ft_myQ = x_all_ft_myQ.tocsr()

        preds = model.predict(x_all_ft_myQ)

        predicted_class = classes[str(preds[0])]

        print(predicted_class)

    QALinks["0"] = predicted_class





    # ### Crawling Quora

    # Set the URL you want to webscrape from
    url = urllib.parse.quote_plus('https://www.quora.com/search?q='+qstn)

    handler = urlopen('https://api.proxycrawl.com/?token=SnUpBD_-K2v7xz_sxCDrHQ&url=' + url)
    
    # Connect to the URL
    response = requests.get('https://api.proxycrawl.com/?token=SnUpBD_-K2v7xz_sxCDrHQ&url=' + url)

    # Parse HTML and save to BeautifulSoup object¶
    soup = BeautifulSoup(response.text, "html5lib")

    table = soup.findAll('div', attrs = {'class':'pagedlist_item'})
    
    # print("Quora")

    for i in range(min(5, len(table))): 
        question = table[i] 
        link = question.find('div', attrs = {'class':'QuestionQueryResult'}).find('a', attrs = {'class':'question_link'})
        document = link['href']
        
        document = document[1:].replace('-', ' ') + '?'
        
        final_link = 'https://www.quora.com/' + link['href']
        
        questions.append(final_link) 
        documents.append(document)

    # ### Crawling StackOverflow

    # Set the URL you want to webscrape from
    url = 'https://stackoverflow.com/search?q=' + qstn

    # Connect to the URL
    response = requests.get(url)

    # Parse HTML and save to BeautifulSoup object¶
    soup = BeautifulSoup(response.text, "html5lib")

    table = soup.findAll('div', attrs = {'class':'question-summary search-result'}) 
    
    # print("StackOverflow")

    for i in range(min(5, len(table))): 
        question = table[i] 
        a_href = question.find('div', attrs = {'class':'summary'}).find('div', attrs = {'class':'result-link'}).h3
        link = a_href.findAll('a')[0]
        document = link['title']
        final_link = 'https://stackoverflow.com/' + link['href']
        
        # print(final_link)
        # print(document)
        
        # print()
        
        questions.append(final_link) 
        documents.append(document)
   
    # ### Crawling Yahoo Answers

    # Set the URL you want to webscrape from
    url = 'https://in.answers.search.yahoo.com/search?q=' + qstn

    # Connect to the URL
    response = requests.get(url)

    # Parse HTML and save to BeautifulSoup object¶
    soup = BeautifulSoup(response.text, "html5lib")

    table = soup.findAll('div', attrs = {'class':'AnswrsV2'}) 
    # print(table)
    
    # print("Yahoo Answers")
    
    for i in range(min(5, len(table))): 
        question = table[i] 
        a_href = question.find('div', attrs = {'class':'compTitle'}).h3
        link = a_href.findAll('a')[0]
        doc = link.contents
        document = []
        for word in doc:
            document.append(BeautifulSoup(str(word).strip(), "lxml").text)
        document = ' '.join(document)
        final_link = link['href']
        
        # print(final_link)
        # print(document)
        
        # print()
        
        questions.append(final_link) 
        documents.append(document)

    # ### Crawling Stack Exchange

    # Set the URL you want to webscrape from
    url = 'https://stackexchange.com/search?q=' + qstn

    # Connect to the URL
    response = requests.get(url)

    # Parse HTML and save to BeautifulSoup object¶
    soup = BeautifulSoup(response.text, "html5lib")

    table = soup.findAll('div', attrs = {'class':'question search-result'}) 
    # print(table)
    
    # print("Stack Exchange")

    for i in range(min(5, len(table))): 
        question = table[i] 
        a_href = question.find('div', attrs = {'class':'result-link'}).span
        link = a_href.findAll('a')[0]
        doc = link.contents
        document = []
        for word in doc:
            document.append(BeautifulSoup(str(word).strip(), "lxml").text)
        document = ' '.join(document)
        final_link = link['href']
        
        # print(final_link)
        # print(document)
        
        # print()
        
        questions.append(final_link) 
        documents.append(document)

    # print("Cosine")

    # ### Finding Cosine Similarity through NLP


    #  Preprocessing with nltk
    # The default functions of CountVectorizer and TfidfVectorizer in scikit-learn detect word
    # boundary and remove punctuations automatically. However, if we want to do
    # stemming or lemmatization, we need to customize certain parameters in
    # CountVectorizer and TfidfVectorizer. Doing this overrides the default tokenization
    # setting, which means that we have to customize tokenization, punctuation removal,
    # and turning terms to lower case altogether.
    # Normalize by stemming: 

	# first-time use only

    # nltk.download('punkt')
    # nltk.download('stopwords')
    # nltk.download('wordnet')

    #Normalize by stemming:

    # nltk.download('punkt') # first-time use only
    stemmer = nltk.stem.porter.PorterStemmer()
    def StemTokens(tokens):
        return [stemmer.stem(token) for token in tokens]

    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

    def StemNormalize(text):
        return StemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict))) 

    # Normalize by lemmatization:

    lemmer = nltk.stem.WordNetLemmatizer()

    def LemTokens(tokens):
        return [lemmer.lemmatize(token) for token in tokens]

    remove_punct_dict = dict((ord(punct), None) for punct in string.punctuation)

    def LemNormalize(text):
        return LemTokens(nltk.word_tokenize(text.lower().translate(remove_punct_dict))) 


    # Turn text into vectors of term frequency:

    stop_words = set(nltk.corpus.stopwords.words('english'))
    tokenized_stop_words = nltk.word_tokenize(' '.join(nltk.corpus.stopwords.words('english')))

    class Tokenizer(object):
        def __init__(self):
            nltk.download('punkt', quiet=True, raise_on_error=True)
            self.stemmer = nltk.stem.PorterStemmer()
            
        def _stem(self, token):
            if (token in stop_words):
                return token  # Solves error "UserWarning: Your stop_words may be inconsistent with your preprocessing."
            return self.stemmer.stem(token)
            
        def __call__(self, line):
            tokens = nltk.word_tokenize(line)
            tokens = (self._stem(token) for token in tokens)  # Stemming
            return list(tokens)

    LemVectorizer = CountVectorizer(tokenizer=Tokenizer(), stop_words = tokenized_stop_words, lowercase=True)
    LemVectorizer.fit_transform(documents)

    # Normalized (after lemmatization) text in the four documents are tokenized and each
    # term is indexed:
    # print (LemVectorizer.vocabulary_ )

    tf_matrix = LemVectorizer.transform(documents).toarray()
    # print (tf_matrix)

    # print(tf_matrix.shape) 

    # Calculate idf and turn tf matrix to tf-idf matrix: 

    # Get idf: 

    tfidfTran = TfidfTransformer(norm="l2")
    tfidfTran.fit(tf_matrix)
    # print (tfidfTran.idf_)

    # Now we have a vector where each component is the idf for each term. In this case, the
    # values are almost the same because other than one term, each term only appears in 1
    # document.

    # Get the tf-idf matrix (4 by 41): 
    tfidf_matrix = tfidfTran.transform(tf_matrix)
    # print (tfidf_matrix.toarray()) 

    # Here what the transform method does is multiplying the tf matrix (4 by 41) by the
    # diagonal idf matrix (41 by 41 with idf for each term on the main diagonal), and dividing
    # the tf-idf by the Euclidean norm. 


    # Get the pairwise similarity matrix (n by n):

    cos_similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
    # print (cos_similarity_matrix)

    # print (cos_similarity_matrix)

    # The matrix obtained in the last step is multiplied by its transpose. The result is the
    # similarity matrix, which indicates that d2 and d3 are more similar to each other than any
    # other pair. 


    # ### Print top 5 question links

    lst = cos_similarity_matrix[0]
    # print(lst)
    top = []

    for j in range(5):
        mx = -1
        index = -1
        for i in range(1, len(documents)):
            if(lst[i] > mx):
                mx = lst[i]
                index = i
        
        top.append(index)
        lst[index] = -1

    # print("Top 5 Questions")

    idx = 0    
    for i in top:
        QALinks[str(idx+1)] = questions[i]
        # print(QALinks[str(idx+1)])
        idx = idx + 1

    response = flask.jsonify(QALinks)
    # print(response)
    response.headers.add('Access-Control-Allow-Origin', '*')
    print(QALinks)
    return response
	
if(__name__ == "__main__"):
    app.run()
