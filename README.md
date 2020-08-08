# Question-Answering-and-Ranking-System
A Web-application based chatbot for answering queries presented by a user through web scraping and Deep Learning methods.

## Introduction

This was a Study Project pursued under Dr Yashvardhan Sharma, CSIS BITS Pilani during 2019. Here, various available Ranking Based Question Answering Systems are reviewed and a technique is proposed which selects the best answer from the available QA models using NLP, and also answers some domain-specific questions which can't be answered by the above systems. A transformer encoder-based approach is proposed for the NLP model.

## Objectives - A Hybrid Search System

The data about private organizations is available to these organizations only and sometimes not available in public domain.

1. To build a context-based Question-Answering System for a specific organization which finds relevant answers to the queries using a corpus of information present only with the organization.

2. Understand and implement how search engines like Google work - Aspects of web crawling, web scraping, ranking and finding relevant answers through a huge web of information.

## Architecture

![Architecture](./Images/Architecture.png?raw=true "Architecture")

## Classes for classification

1. *Society & Culture* - e.g. what are the social views of people from different cultures on abstinence? 
2. *Science & Mathematics* - e.g. What is fringe in optics? 
3. *Health* - e.g. What the hell is restless leg syndrome? 
4. *Education & Reference* - e.g. what is the past tense of tell? 
5. *Computers & Internet* - e.g. Do You have the code for the Luxor game? 
6. *Sports* - e.g. Who is going to win the FIFA World CUP 2006 in Germany? 
7. *Business & Finance* - e.g. What is secretary as corporation director? 
8. *Entertainment & Music* - e.g. where can I download mp3 songs? 
9. *Family & Relationships* - e.g. who's ready for Christmas? 
10. *Politics & Government* - e.g. Isn't civil war and oxymoron?


## Dataset

We used Yahoo! Answers topic selection dataset
  - Human labelled dataset constructed with 10 largest main categories
  - Each class contains 1,40,000 training and 6,000 testing samples

From all answers and other meta information, only the best answer content and the main category information were used.

## Proposed Technique

The architecture of the system comprises of 4 modules:

1. Question Classificatier
2. Question Answering System
3. Question Selection Web Service
4. Chrome/Firefox Extension

## Implementation Details

1. Tokenization
2. Stop words removal 
3. Lemmatizing with NLTK 
4. Measuring the Cosine Similarity 

## Question Classification - Approach

1. Text Exploration
2. Text Cleaning
3. Obtaining POS Tags, Identifying Named Entities, Lemmas, Syntactic Dependency Relations and Orthographic Features.
4. Using the obtained properties as Features.
5. Using a Linear SVM model on the engineered features.

## Model

Linear Support Vector Machine Classifier

**Features used:** 
  - Named Entity Recognition
  - Lemmas
  - POS Tags

Accuracy: `66.316%`

## Implementation

![Question_Search](./Images/Question_Search.png?raw=true "Question_Search")

## Context-Based Classification - BERT

*BERT:* Bidirectional Encoder Representations from Transformers

*Transformers:* Models that process words in relation to all the other words in a sentence, rather than one-by-one in order. BERT models can therefore consider the full context of a word by looking at the words that come before and after it—particularly useful for understanding the intent behind search queries.

## How BERT works?

As opposed to directional models, which read the text input sequentially (left-to-right or right-to-left), the Transformer encoder reads the entire sequence of words at once.

Therefore it is considered bidirectional, though it would be more accurate to say that it is non-directional. This characteristic allows the model to learn the context of a word based on all of its surroundings (left and right of the word).

## Masked LM (MLM)

Before feeding word sequences into BERT, 15% of the words in each sequence are replaced with a `[MASK]` token. The model then attempts to predict the original value of the masked words, based on the context provided by the other, non-masked, words in the sequence. 

In technical terms, the prediction of the output words requires:
1. Adding a classification layer on top of the encoder output.
2. Multiplying the output vectors by the embedding matrix, transforming them into the vocabulary dimension.
3. Calculating the probability of each word in the vocabulary with softmax.

![MLM](./Images/MLM.png?raw=true "MLM")

## Fine-tuning BERT for Q&A Task

In Question Answering tasks the software receives a question regarding a text sequence and is required to mark the answer in the sequence. Using BERT, a Q&A model can be trained by learning two extra vectors that mark the beginning and the end of the answer.

### BERT Input Format

![BERT_Input_Format](./Images/BERT_Input_Format.png?raw=true "BERT_Input_Format")

### Start Token Classifier

![Start_Token_Classifier](./Images/Start_Token_Classifier.png?raw=true "Start_Token_Classifier")

### End Token Classifier

![End_Token_Classifier](./Images/End_Token_Classifier.png?raw=true "End_Token_Classifier")


## Multilingual Application

![Multilingual_App](./Images/Multilingual_App.png?raw=true "Multilingual_App")


## Installing the Chrome Extension from Source

1. Clone this repo so you have a copy in a folder locally.
2. Open `chrome://extensions` in the location or go to `Tools` > `Extensions`
3. Enable `Developer mode` by checking the checkbox in the upper-right corner.
4. Click on the button labelled `Load unpacked extension...`.
5. Select the directory where you cloned this repo to.

