# -*- coding: utf-8 -*-
"""
Created on Tue Aug 25 13:21:23 2020

@author: Nachiketa Dhal
"""


# importing the Dataset

import pandas as pd

messages = pd.read_csv('Resources/SMSSpamCollection', sep='\t',
                           names=["label", "message"])

#Data cleaning and preprocessing
import re
import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
ps = PorterStemmer()
lemma = WordNetLemmatizer()
corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])   # Removes ,.<>? etc.. 
    review = review.lower()
    review = review.split()
    
    review = [lemma.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
# tfidf = TfidfVectorizer(max_features=5000)
cv = CountVectorizer(max_features=5000) # Takes 5000 columns
X = cv.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['label'])     # Creates 2 column named spam or ham
y = y.iloc[:,-1]  # Drops one column(takes only last column i.e spam)


# Train Test Split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)

# Training model using Naive bayes classifier

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)

y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy_score = accuracy_score(y_test, y_pred) 
