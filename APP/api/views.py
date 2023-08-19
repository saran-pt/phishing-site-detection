import os
import pickle
import pandas as pd

from nltk.tokenize import RegexpTokenizer 
from nltk.stem.snowball import SnowballStemmer

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.pipeline import make_pipeline

from rest_framework.views import APIView
from rest_framework.response import Response
from api.models import Urls
from .seriailzers import UrlSerializer


def get_phishing_data():
    """
    Process and clean the url's data
    :parm: None
    :return: List of tokenized data
    """
    print('Data preprocessing initiated......')
    phishing_data = pd.read_csv('../phishing_site_urls.csv')
    tokenizer = RegexpTokenizer(r'[A-Za-z]+')
    phishing_data['text_tokenized'] = phishing_data.URL.map(lambda url: tokenizer.tokenize(url))
    
    stemmer = SnowballStemmer('english')
    phishing_data['text_stemmed'] = phishing_data.text_tokenized.map(lambda words: [stemmer.stem(word) for word in words])

    phishing_data['text_sent'] = phishing_data.text_stemmed.map(lambda words: ' '.join(words))

    bad_sites = phishing_data[phishing_data.Label == 'bad']   
    bad_sites_data = bad_sites.text_sent
    bad_sites_data.reset_index(drop=True, inplace=True)

    good_sites = phishing_data[phishing_data.Label == 'good']
    good_sites_data = good_sites.text_sent
    good_sites_data.reset_index(drop=True, inplace=True)

    return phishing_data


def train_model():
    """
    Load the trained model in a pickle file
    :parm: None
    :return: None
    """
    phishing_data = get_phishing_data()

    pipeline_ls = make_pipeline(CountVectorizer(tokenizer = RegexpTokenizer(r'[A-Za-z]+').tokenize,stop_words='english'), LogisticRegression())
    
    print('Model training begins.......')
    trainX, testX, trainY, testY = train_test_split(phishing_data.URL, phishing_data.Label)
    pipeline_ls.fit(trainX, trainY)

    print(pipeline_ls.score(testX, testY))
    pickle.dump(pipeline_ls, open('../phishing.pkl', 'wb'))


class HomeView(APIView):
    serializer = UrlSerializer

    def post(self, request):
        url = request.data['url']
        file_name : str = '../phishing.pkl'
        if not os.path.isfile(file_name):
            print('Pkl file not found. Training new model!')
            train_model()

        loaded_model = pickle.load(open(file_name, 'rb'))
        data = loaded_model.predict([url])[0]

        return Response({"message":f'{data.upper()}'})
    