from sklearn.model_selection import train_test_split
import numpy as np
import sklearn as sk
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer


embs = np.load('./emb.npy', allow_pickle=True)
labels = np.load('./label.npy', allow_pickle=True)
print(embs.shape)
print(labels.shape)
# test_emb = np.load('.npy')
# test_label = np.load('.npy')

X_train, X_test, Y_train, Y_test = train_test_split(embs, labels, test_size=0.33, random_state=321)
print("ss : ",X_train.shape, Y_train.shape)
print("test : ",X_test.shape, Y_test.shape)
svc_1 = SVC(kernel='linear')


in_encoder = Normalizer(norm='l2')
X_train = in_encoder.transform(X_train)
X_test = in_encoder.transform(X_test)

out_encoder = LabelEncoder()
out_encoder.fit(Y_train)
Y_train = out_encoder.transform(Y_train)
Y_test = out_encoder.transform(Y_test)

def train_and_evaluate(clf,X_train,X_test,Y_train,Y_test):
    
    clf.fit(X_train,Y_train)
    
    print ("Accuracy on training set:")
    print (clf.score(X_train, Y_train))
    print ("Accuracy on testing set:")
    print (clf.score(X_test, Y_test))
    
    Y_pred=clf.predict(X_test)
    
    print ("Classification Report:")
    print (metrics.classification_report(Y_test, Y_pred))
    print ("Confusion Matrix:")
    print (metrics.confusion_matrix(Y_test, Y_pred))

train_and_evaluate(svc_1,X_train,X_test,Y_train,Y_test)

joblib.dump(svc_1, 'svc_face.pkl')

# x = embData
# #y = labelData
svc = joblib.load('svc_face.pkl')
# svc.predict(x)

