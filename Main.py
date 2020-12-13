# -*- coding: utf-8 -*-
"""
Created on Sat Dec 12 01:46:17 2020

Sentiment Labelled Sentences Data Set

Data Set Information:

This dataset was created for the Paper 'From Group to Individual Labels using Deep Features', Kotzias et. al,. KDD 2015
Please cite the paper if you want to use it :)

It contains sentences labelled with positive or negative sentiment.

=======
Format:
=======
sentence score


=======
Details:
=======
Score is either 1 (for positive) or 0 (for negative)
The sentences come from three different websites/fields:

imdb.com
amazon.com
yelp.com

For each website, there exist 500 positive and 500 negative sentences. Those were selected randomly for larger datasets of reviews.
We attempted to select sentences that have a clearly positive or negative connotaton, the goal was for no neutral sentences to be selected.

@author: obemb
"""

# Step 1

# Import libraries
import pandas as pd
import numpy as np
print("libraries installed succesffully!")


# Acquire dataset/datframe
title = ['Review', 'Label']

df1 = pd.read_csv('amazon_cells_labelled.txt', delimiter = '\t',  names= title,  quoting= 3, engine = 'python', encoding = 'latin-1')

df2 = pd.read_csv('imdb_labelled.txt', quoting = 3,  delimiter = '\t',  sep = '.' , names = title, engine = 'python', encoding = 'latin-1')

df3 = pd.read_csv('yelp_labelled.txt', delimiter = '\t', sep = '.' ,  names = title, quoting = 3, engine = 'python', encoding = 'latin-1')

df = pd.concat([df1, df2, df3], axis = 0, sort=False, ignore_index= True)
#df = pd.concat([df_a, df3], axis = 0, sort=False, ignore_index= True)
df.reset_index(drop = True)

print(" The shape of the dataframe is: " , df.shape)

# Step 2: Text Cleaning

import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

for i in range( df.shape[0]):
    review = re.sub('[^a-zA-Z]', ' ', df['Review'][i] )
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    all_stopwords = stopwords.words('english')
    all_stopwords.remove("isn't")
    all_stopwords.remove("wasn't")
    all_stopwords.remove("not")
    review = [ps.stem(word) for word in review 
              if not word in set(all_stopwords)]
    review = ' '.join(review)
    corpus.append(review)
    
#print(corpus)

# Step 3 : Create Bag of Word Model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=3500)
X = cv.fit_transform(corpus).toarray()
y = df.iloc[:,-1].values

print(len(X[0]))


# Step 3  Split data to training and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42)

# Train Model with Naive Bayes algorithm

# Training the Naive Bayes model on the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB( )
classifier.fit(X_train, y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
score = accuracy_score(y_test, y_pred)

print("The accuracy for the NB Model is :", score)