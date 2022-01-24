# [1]
import pandas as pd
msg = pd.read_csv('Datasets/document.csv', names=['message', 'label'])
print("Total Instances of Dataset: ", msg.shape[0])
msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

# [2]
X = msg.message
y = msg.labelnum
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y)
count_v = CountVectorizer()
X_train_dm = count_v.fit_transform(X_train)
X_test_dm = count_v.transform(X_test)

# [3]
df = pd.DataFrame(X_train_dm.toarray(), columns=count_v.get_feature_names())

# [4]
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB()
clf.fit(X_train_dm, y_train)
pred = clf.predict(X_test_dm)
for doc, p in zip(X_train, pred):
    p = 'pos' if p == 1 else 'neg'
    print(f"{doc} -> {p}")

# [5]
from sklearn.metrics import (accuracy_score, confusion_matrix, precision_score,recall_score)

print('Accuracy Metrics: \n')
print('Accuracy: ', accuracy_score(y_test, pred))
print('Recall: ', recall_score(y_test, pred))
print('Precision: ', precision_score(y_test, pred))
print('Confusion Matrix: \n', confusion_matrix(y_test, pred))