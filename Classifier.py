import sys
from time import time
sys.path.append("../Mail_Classifier/")
from Vectorizer import Vectorizer

sys.path.append("../Mail_Classifier/")
from New_Mail_Process import NVectorizer

features_train, features_test, labels_train, labels_test = Vectorizer()

features_new = NVectorizer()

from sklearn import svm
clf = svm.SVC(C=1, kernel='linear')


t0 = time()
clf.fit(features_train, labels_train)
print "Training time:", round(time()-t0, 6), "s"

from sklearn.metrics import accuracy_score
pred1 = clf.predict(features_train)
pred2 = clf.predict(features_test)
print "Training Accuracy = ", accuracy_score(labels_train, pred1)
print "Testing Accuracy = ", accuracy_score(labels_test,pred2)

print "Category of new mail : ", clf.predict(features_new)
