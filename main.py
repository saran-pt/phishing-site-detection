import os
import pickle
import numpy as np 
import pandas as pd

from sklearn.linear_model import LogisticRegression # algo use to predict good or bad
from sklearn.naive_bayes import MultinomialNB # nlp algo use to predict good or bad

from sklearn.model_selection import train_test_split # spliting the data between feature and target
from sklearn.metrics import classification_report # gives whole report about metrics (e.g, recall,precision,f1_score,c_m)
from sklearn.metrics import confusion_matrix # gives info about actual and predict
from nltk.tokenize import RegexpTokenizer # regexp tokenizers use to split words from text  
from nltk.stem.snowball import SnowballStemmer # stemmes words
from sklearn.feature_extraction.text import CountVectorizer # create sparse matrix of words using regexptokenizes  
from sklearn.pipeline import make_pipeline # use for combining all prerocessors techniuqes and algos


phish_data = pd.read_csv('../phishing_site_urls.csv')
tokenizer = RegexpTokenizer(r'[A-Za-z]+')#to getting alpha only
phish_data['text_tokenized'] = phish_data.URL.map(lambda t: tokenizer.tokenize(t)) # doing with all rows

stemmer = SnowballStemmer("english") # choose a language
phish_data['text_stemmed'] = phish_data['text_tokenized'].map(lambda l: [stemmer.stem(word) for word in l])

phish_data['text_sent'] = phish_data['text_stemmed'].map(lambda l: ' '.join(l))

#sliceing classes
bad_sites = phish_data[phish_data.Label == 'bad']
good_sites = phish_data[phish_data.Label == 'good']

data = good_sites.text_sent
data.reset_index(drop=True, inplace=True)
common_text = str(data)

data = bad_sites.text_sent
data.reset_index(drop=True, inplace=True)
common_text = str(data)

#create cv object
cv = CountVectorizer()

feature = cv.fit_transform(phish_data.text_sent) #transform all text which we tokenize and stemed

trainX, testX, trainY, testY = train_test_split(feature, phish_data.Label)

# create lr object
lr = LogisticRegression()

lr.fit(trainX,trainY)

Scores_ml = {}
Scores_ml['Logistic Regression'] = np.round(lr.score(testX,testY),2)


print('Training Accuracy :',lr.score(trainX,trainY))
print('Testing Accuracy :',lr.score(testX,testY))
con_mat = pd.DataFrame(confusion_matrix(lr.predict(testX), testY),
            columns = ['Predicted:Bad', 'Predicted:Good'],
            index = ['Actual:Bad', 'Actual:Good'])

print('\nCLASSIFICATION REPORT\n')
print(classification_report(lr.predict(testX), testY, target_names =['Bad','Good']))

pickle.dump(pipeline_ls,open('phishing.pkl','wb'))

loaded_model = pickle.load(open('phishing.pkl', 'rb'))

test_url = input()
print(loaded_model.predict([test_url]))
