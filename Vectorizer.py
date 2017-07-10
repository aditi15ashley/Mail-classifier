import os
import sys
import numpy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV

def Vectorizer() :
	sys.path.append( "../Mail_Classifier/" )
	from Parsing import main

	#main()

	f = open("../Mail_Classifier/Parsedtext.txt", "r")

	g =open("../Mail_Classifier/Train_Author.txt", "r")

	vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.6, stop_words='english')

	processed_input = vectorizer.fit_transform(f)
	h = g.read()
	processed_labels = h.split()

	processed_input, processed_labels = shuffle(processed_input, processed_labels, random_state=0)

	feature_train1, feature_validation, labels_train1, labels_validation = train_test_split(processed_input, processed_labels, test_size = 0.2, random_state = 42)
	feature_train, feature_test, labels_train, labels_test = train_test_split(feature_train1, labels_train1, test_size = 0.25, random_state = 42)

	#parameters = {'kernel':('linear', 'rbf'), 'C':[1,100000000]}
	#from sklearn.svm import SVC
	#clf = SVC()
	#cv = GridSearchCV(clf,parameters)
	#cv.fit(feature_validation, labels_validation)
	#dict = cv.best_params_
	#print "C Value =",dict['C']
	#print "Kernel =",dict['kernel']

	return feature_train, feature_test, labels_train, labels_test
