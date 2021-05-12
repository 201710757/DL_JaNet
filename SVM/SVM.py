from sklearn.model_selection import train_test_split
import numpy as np
import sklearn as sk
from sklearn.svm import SVC

embs = np.load('.npy')
labels = np.load('.npy')

# test_emb = np.load('.npy')
# test_label = np.load('.npy')

X_train, X_test, Y_train, Y_test = train_test_split(embs, labels, test_size=0.33, random_state=321)

svc_1 = SVC(kernel='linear')

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
svc.predict(x)

